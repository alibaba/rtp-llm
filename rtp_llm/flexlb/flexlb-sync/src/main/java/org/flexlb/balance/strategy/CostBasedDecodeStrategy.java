package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

@Component
public class CostBasedDecodeStrategy implements LoadBalanceStrategy {

    private static final int SNAPSHOT_CAPTURE_ATTEMPTS = 2;

    private final WorkerDirectory workerDirectory;
    private final DecodeResourceMeasure resourceMeasure;

    public CostBasedDecodeStrategy(WorkerDirectory workerDirectory,
                                    DecodeResourceMeasure resourceMeasure) {
        this.workerDirectory = workerDirectory;
        this.resourceMeasure = resourceMeasure;
    }

    @Override
    public boolean supports(
            RoleType role,
            RoutingConfig.EndpointSelectorConfig configured) {
        return role == RoleType.DECODE
                && configured
                instanceof RoutingConfig.KvUsageWeightedRandomConfig;
    }

    @Override
    public SelectedRole select(BalanceContext balanceContext, RoleType roleType, String group) {
        Request request = balanceContext.getRequest();
        long seqLen = request.getSeqLen();
        FlexlbConfig config = balanceContext.getConfig();
        RoutingConfig.KvUsageWeightedRandomConfig selector =
                (RoutingConfig.KvUsageWeightedRandomConfig) config.getRouter().getRoles()
                        .getDecode().getSelector();

        // Queues without preemption retain transient Decode pressure at the
        // exact pre-delivery permit. Only preemptive placement needs a miss here
        // in order to enter victim planning.
        boolean softQueuePlacement =
                config.defersDecodeCapacityUntilDispatch();
        for (int attempt = 0; attempt < SNAPSHOT_CAPTURE_ATTEMPTS; attempt++) {
            SnapshotSelection selection = selectFromSnapshots(
                    balanceContext,
                    roleType,
                    group,
                    seqLen,
                    selector,
                    softQueuePlacement);
            if (selection.selected() != null) {
                return selection.selected();
            }
            if (!selection.captureConflict()) {
                return null;
            }
        }

        Logger.debug(
                "Decode snapshot winner changed repeatedly; retry placement,"
                    + " request_id={}",
                balanceContext.getRequestId());
        return null;
    }

    @Override
    public EndpointSelection selectForQueue(
            BalanceContext context, RoleType roleType, String group) {
        SelectedRole selected = select(context, roleType, group);
        if (selected != null) {
            return EndpointSelection.selected(selected);
        }
        return EndpointSelection.unavailable(roleType);
    }

    private SnapshotSelection selectFromSnapshots(
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            boolean softQueuePlacement) {
        EndpointFilterResult<EndpointRegistry.DecodeRoutingSnapshot> filterResult =
                getAvailableEndpointSnapshots(
                        group, softQueuePlacement, seqLen);
        List<EndpointRegistry.DecodeRoutingSnapshot> eligible =
                filterResult.candidates();
        if (eligible.isEmpty()) {
            logNoAvailableEndpoint(balanceContext, filterResult);
            return SnapshotSelection.complete(null);
        }

        PreferredSelection<EndpointRegistry.DecodeRoutingSnapshot> selection =
                selectPreferredThenFallback(
                        eligible, SNAPSHOT_ROUTING, balanceContext, selector,
                        softQueuePlacement);
        EndpointRegistry.DecodeRoutingSnapshot selected = selection.selected();
        if (selected == null) {
            logAllFilteredOut(
                    balanceContext,
                    filterResult.rejections(),
                    selection.filters().rejections());
            return SnapshotSelection.complete(null);
        }

        WorkerEndpoint.GenerationPin pin =
                workerDirectory.captureDecodeGeneration(
                        selected);
        if (pin == null) {
            return SnapshotSelection.conflict();
        }
        return SnapshotSelection.complete(buildSelectedRole(
                selected, pin, roleType, balanceContext));
    }

    private static final Function<EndpointRegistry.DecodeRoutingSnapshot,
            DecodeEndpoint.DecodeRoutingView>
            SNAPSHOT_ROUTING = EndpointRegistry.DecodeRoutingSnapshot::routing;

    private record EndpointFilterResult<T>(
            List<T> candidates,
            Map<String, Integer> rejections,
            int registered) {}

    private record FilterResult<T>(
            List<T> candidates,
            Map<String, Integer> rejections,
            long minCacheUsed,
            long maxCacheUsed,
            int minCacheUsedIndex,
            boolean allSameUsage) {}

    private record PreferredSelection<T>(
            T selected,
            FilterResult<T> filters) {}

    private record SnapshotSelection(
            SelectedRole selected,
            boolean captureConflict) {
        private static SnapshotSelection complete(SelectedRole selected) {
            return new SnapshotSelection(selected, false);
        }

        private static SnapshotSelection conflict() {
            return new SnapshotSelection(null, true);
        }
    }

    private EndpointFilterResult<EndpointRegistry.DecodeRoutingSnapshot>
            getAvailableEndpointSnapshots(
            String group,
            boolean softQueuePlacement,
            long seqLen) {
        DecodeResourceMeasure measure = resourceMeasure;
        List<EndpointRegistry.DecodeRoutingSnapshot> snapshots =
                workerDirectory.decodeRoutingSnapshot(group);
        int registered = snapshots.size();
        if (registered == 0) {
            return new EndpointFilterResult<>(
                    snapshots, Map.of("NO_REGISTERED", 1), 0);
        }
        rejectIfPhysicalCapacityIsTooSmall(snapshots, seqLen);
        if (softQueuePlacement) {
            return new EndpointFilterResult<>(
                    snapshots, Map.of(), registered);
        }

        ArrayList<EndpointRegistry.DecodeRoutingSnapshot> filtered = null;
        int unavailable = 0;
        for (int index = 0; index < registered; index++) {
            EndpointRegistry.DecodeRoutingSnapshot snapshot =
                    snapshots.get(index);
            if (!measure.isResourceAvailable(snapshot.routing())) {
                unavailable++;
                if (filtered == null) {
                    filtered = new ArrayList<>(registered);
                    filtered.addAll(snapshots.subList(0, index));
                }
                continue;
            }
            if (filtered != null) {
                filtered.add(snapshot);
            }
        }

        if (filtered == null) {
            return new EndpointFilterResult<>(
                    snapshots, Map.of(), registered);
        }
        return new EndpointFilterResult<>(
                Collections.unmodifiableList(filtered),
                Map.of("RESOURCE_UNAVAILABLE", unavailable),
                registered);
    }

    private static void rejectIfPhysicalCapacityIsTooSmall(
            List<EndpointRegistry.DecodeRoutingSnapshot> endpoints,
            long requiredKv) {
        long maximum = 0L;
        for (EndpointRegistry.DecodeRoutingSnapshot endpoint : endpoints) {
            long physicalKv = endpoint.routing().totalKv();
            if (physicalKv <= 0L) {
                return;
            }
            maximum = Math.max(maximum, physicalKv);
        }
        if (Math.max(0L, requiredKv) > maximum) {
            throw new StaticCapacityExceededException(
                    "Decode request seq_len=" + requiredKv
                            + " exceeds max known physical KV=" + maximum);
        }
    }

    /**
     * Prefer endpoints that can dispatch immediately. If every endpoint is
     * temporarily full, retain the complete eligible set so QUEUE placement
     * can wait for the next exact dispatch-capacity signal.
     */
    private <T> PreferredSelection<T> selectPreferredThenFallback(
            List<T> eligible,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing,
            BalanceContext context,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            boolean softQueuePlacement) {
        long seqLen = context.getRequest().getSeqLen();
        List<T> dispatchable = softQueuePlacement
                ? dispatchableForRequest(eligible, routing, context)
                : eligible;
        List<T> projected = softQueuePlacement
                ? leastProjectedOwnership(dispatchable, routing)
                : dispatchable;
        PreferredSelection<T> selection = selectWithin(
                projected, routing, seqLen, selector, softQueuePlacement);
        if (selection.selected() != null) {
            return selection;
        }
        if (projected.size() != dispatchable.size()) {
            selection = selectWithin(
                    dispatchable, routing, seqLen, selector,
                    softQueuePlacement);
            if (selection.selected() != null) {
                return selection;
            }
        }
        if (dispatchable.size() != eligible.size()) {
            return selectWithin(
                    eligible, routing, seqLen, selector,
                    softQueuePlacement);
        }
        return selection;
    }

    private <T> PreferredSelection<T> selectWithin(
            List<T> candidates,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing,
            long seqLen,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            boolean softQueuePlacement) {
        FilterResult<T> filters = applyHardFilters(
                candidates, routing, seqLen, selector, softQueuePlacement);
        return new PreferredSelection<>(
                weightedRandomSelection(
                        filters, routing, selector.getDecayPerToken()),
                filters);
    }

    /** Return the original list when all or no endpoints fit this request. */
    private static <T> List<T> dispatchableForRequest(
            List<T> eligible,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing,
            BalanceContext context) {
        ArrayList<T> available = new ArrayList<>(eligible.size());
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
        for (T candidate : eligible) {
            DecodeEndpoint.DecodeRoutingView view = routing.apply(candidate);
            long expectedKv = Math.max(
                    hardKv,
                    context.getConfig().decodeKvReservationTokens(
                            hardKv, maxNewTokens, view.totalKv()));
            DecodeEndpoint.EngineDispatchDemand demand =
                    new DecodeEndpoint.EngineDispatchDemand(hardKv, expectedKv);
            if (DecodeEndpoint.canAcquireEngineDispatchPermit(
                    view, demand, capacity)) {
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
    private static <T> List<T> leastProjectedOwnership(
            List<T> candidates,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing) {
        int minimum = Integer.MAX_VALUE;
        ArrayList<T> leastOwned = new ArrayList<>(candidates.size());
        for (T candidate : candidates) {
            int projected = routing.apply(candidate).totalLoad();
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

    private <T> FilterResult<T> applyHardFilters(
            List<T> eligible,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing,
            long seqLen,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            boolean softQueuePlacement) {
        RoutingConfig.DecodeOutlierRejectionConfig outlier = selector.getOutlierRejection();
        double hotspotMultiplier = outlier == null
                ? 0.0 : outlier.getMaxEngineLoadVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0 : outlier.getMaxKvUsedVsAverageMultiplier();

        int n = eligible.size();
        long sumLoad = 0;
        long sumCacheUsed = 0;
        ArrayList<T> capacitySurvivors = null;
        int capacityRejected = 0;
        int capacitySurvivorCount = 0;
        long firstCapacityCacheUsed = 0;
        long minCapacityCacheUsed = 0;
        long maxCapacityCacheUsed = 0;
        int minCapacityCacheUsedIndex = 0;
        boolean allCapacityUsageSame = true;
        for (int index = 0; index < n; index++) {
            T candidate = eligible.get(index);
            DecodeEndpoint.DecodeRoutingView view = routing.apply(candidate);
            sumLoad += view.engineLoad();
            long cacheUsed = view.realKvUsed();
            sumCacheUsed += cacheUsed;

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
            if (capacitySurvivorCount == 0) {
                firstCapacityCacheUsed = cacheUsed;
                minCapacityCacheUsed = cacheUsed;
                maxCapacityCacheUsed = cacheUsed;
            } else {
                if (cacheUsed != firstCapacityCacheUsed) {
                    allCapacityUsageSame = false;
                }
                if (cacheUsed < minCapacityCacheUsed) {
                    minCapacityCacheUsed = cacheUsed;
                    minCapacityCacheUsedIndex = capacitySurvivorCount;
                }
                if (cacheUsed > maxCapacityCacheUsed) {
                    maxCapacityCacheUsed = cacheUsed;
                }
            }
            capacitySurvivorCount++;
        }
        long avgLoad = sumLoad / n;
        long avgCacheUsed = sumCacheUsed / n;

        List<T> capacityCandidates = capacitySurvivors == null
                ? eligible
                : Collections.unmodifiableList(capacitySurvivors);
        boolean filterHotspots = hotspotMultiplier > 0 && avgLoad > 0;
        boolean filterImbalance = imbalanceMultiplier > 0 && avgCacheUsed > 0;
        if (capacitySurvivorCount == 0
                || (!filterHotspots && !filterImbalance)) {
            return new FilterResult<>(
                    capacityCandidates,
                    hardFilterRejections(capacityRejected, 0, 0),
                    minCapacityCacheUsed,
                    maxCapacityCacheUsed,
                    minCapacityCacheUsedIndex,
                    allCapacityUsageSame);
        }

        ArrayList<T> outlierSurvivors = null;
        int hotspotRejected = 0;
        int imbalanceRejected = 0;
        int outlierSurvivorCount = 0;
        long firstOutlierCacheUsed = 0;
        long minOutlierCacheUsed = 0;
        long maxOutlierCacheUsed = 0;
        int minOutlierCacheUsedIndex = 0;
        boolean allOutlierUsageSame = true;
        for (int index = 0; index < capacitySurvivorCount; index++) {
            T candidate = capacityCandidates.get(index);
            DecodeEndpoint.DecodeRoutingView view = routing.apply(candidate);
            if (filterHotspots
                    && view.engineLoad() > avgLoad * hotspotMultiplier) {
                hotspotRejected++;
                if (outlierSurvivors == null) {
                    outlierSurvivors = new ArrayList<>(capacitySurvivorCount);
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
                    outlierSurvivors = new ArrayList<>(capacitySurvivorCount);
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
            if (outlierSurvivorCount == 0) {
                firstOutlierCacheUsed = cacheUsed;
                minOutlierCacheUsed = cacheUsed;
                maxOutlierCacheUsed = cacheUsed;
            } else {
                if (cacheUsed != firstOutlierCacheUsed) {
                    allOutlierUsageSame = false;
                }
                if (cacheUsed < minOutlierCacheUsed) {
                    minOutlierCacheUsed = cacheUsed;
                    minOutlierCacheUsedIndex = outlierSurvivorCount;
                }
                if (cacheUsed > maxOutlierCacheUsed) {
                    maxOutlierCacheUsed = cacheUsed;
                }
            }
            outlierSurvivorCount++;
        }

        if (outlierSurvivors == null) {
            return new FilterResult<>(
                    capacityCandidates,
                    hardFilterRejections(capacityRejected, 0, 0),
                    minCapacityCacheUsed,
                    maxCapacityCacheUsed,
                    minCapacityCacheUsedIndex,
                    allCapacityUsageSame);
        }
        return new FilterResult<>(
                Collections.unmodifiableList(outlierSurvivors),
                hardFilterRejections(
                        capacityRejected,
                        hotspotRejected,
                        imbalanceRejected),
                minOutlierCacheUsed,
                maxOutlierCacheUsed,
                minOutlierCacheUsedIndex,
                allOutlierUsageSame);
    }

    private static Map<String, Integer> hardFilterRejections(
            int capacityRejected,
            int hotspotRejected,
            int imbalanceRejected) {
        if (capacityRejected == 0
                && hotspotRejected == 0
                && imbalanceRejected == 0) {
            return Map.of();
        }
        if (hotspotRejected == 0 && imbalanceRejected == 0) {
            return Map.of("KV_CAPACITY", capacityRejected);
        }
        if (capacityRejected == 0 && imbalanceRejected == 0) {
            return Map.of("HOTSPOT_FILTERED", hotspotRejected);
        }
        if (capacityRejected == 0 && hotspotRejected == 0) {
            return Map.of("IMBALANCE_FILTERED", imbalanceRejected);
        }
        if (capacityRejected == 0) {
            return Map.of(
                    "HOTSPOT_FILTERED", hotspotRejected,
                    "IMBALANCE_FILTERED", imbalanceRejected);
        }
        if (hotspotRejected == 0) {
            return Map.of(
                    "KV_CAPACITY", capacityRejected,
                    "IMBALANCE_FILTERED", imbalanceRejected);
        }
        if (imbalanceRejected == 0) {
            return Map.of(
                    "KV_CAPACITY", capacityRejected,
                    "HOTSPOT_FILTERED", hotspotRejected);
        }
        return Map.of(
                "KV_CAPACITY", capacityRejected,
                "HOTSPOT_FILTERED", hotspotRejected,
                "IMBALANCE_FILTERED", imbalanceRejected);
    }

    private <T> T weightedRandomSelection(
            FilterResult<T> filtered,
            Function<T, DecodeEndpoint.DecodeRoutingView> routing,
            double decayFactor) {
        List<T> candidates = filtered.candidates();
        if (candidates.isEmpty()) {
            return null;
        }

        int n = candidates.size();

        if (filtered.allSameUsage() || decayFactor == 0.0) {
            return candidates.get(ThreadLocalRandom.current().nextInt(n));
        }

        double[] weights = new double[n];
        double totalWeight = 0;
        // Subtract the value that produces the maximum log-weight before exponentiation.
        // This is mathematically equivalent to the previous average-centered weights, but
        // keeps every exponent <= 0 and avoids exp(...) overflowing for large KV gaps.
        long referenceCacheUsed = decayFactor >= 0
                ? filtered.minCacheUsed()
                : filtered.maxCacheUsed();
        for (int i = 0; i < n; i++) {
            long cacheUsed = routing.apply(candidates.get(i)).realKvUsed();
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
            return candidates.get(filtered.minCacheUsedIndex());
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
        return candidates.get(filtered.minCacheUsedIndex());
    }

    private SelectedRole buildSelectedRole(
            EndpointRegistry.DecodeRoutingSnapshot selected,
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
            DecodeEndpoint.DecodeRoutingView routing = selected.routing();
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
            EndpointFilterResult<?> filterResult) {
        Logger.debug(
                "Decode select failed: no available endpoints, request_id={}, registered={},"
                    + " eligible=0, rejections={}",
                balanceContext.getRequestId(),
                filterResult.registered(),
                filterResult.rejections());
    }

    private static void logAllFilteredOut(
            BalanceContext balanceContext,
            Map<String, Integer> availabilityRejections,
            Map<String, Integer> hardRejections) {
        Map<String, Integer> merged =
                new java.util.HashMap<>(availabilityRejections);
        hardRejections.forEach(
                (key, count) -> merged.merge(key, count, Integer::sum));
        Logger.debug(
                "Decode select failed: all filtered out, request_id={}, rejections={}",
                balanceContext.getRequestId(), merged);
    }
}
