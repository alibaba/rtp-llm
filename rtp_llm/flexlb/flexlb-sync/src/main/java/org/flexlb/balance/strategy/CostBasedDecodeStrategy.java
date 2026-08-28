package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

@Component
public class CostBasedDecodeStrategy implements LoadBalanceStrategy {

    private static final int SNAPSHOT_CAPTURE_ATTEMPTS = 2;

    private final EngineWorkerStatus engineWorkerStatus;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public CostBasedDecodeStrategy(EngineWorkerStatus engineWorkerStatus,
                                    ResourceMeasureFactory resourceMeasureFactory) {
        this.engineWorkerStatus = engineWorkerStatus;
        this.resourceMeasureFactory = resourceMeasureFactory;
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

        // FIFO queues may wait for transient Decode pressure to drain. A
        // PRIORITY queue deliberately retains the inclusive KV gate because
        // route failure is what enters its typed admission/preemption path.
        boolean softQueuePlacement = config.isQueue() && !config.isPriorityOrdering();
        ResourceMeasureIndicatorEnum indicator =
                config.resourceMeasureFor(roleType);
        for (int attempt = 0; attempt < SNAPSHOT_CAPTURE_ATTEMPTS; attempt++) {
            SnapshotSelection selection = selectFromSnapshots(
                    balanceContext,
                    roleType,
                    group,
                    seqLen,
                    selector,
                    indicator,
                    softQueuePlacement);
            if (selection.selected() != null) {
                return selection.selected();
            }
            if (!selection.captureConflict()) {
                return null;
            }
        }

        // Repeated winner-version conflicts are unusual. Preserve the former
        // liveness semantics by falling back to the exact all-generation pin
        // path instead of rejecting an otherwise serviceable request.
        Logger.debug(
                "Decode snapshot winner changed repeatedly; using pinned fallback,"
                    + " request_id={}",
                balanceContext.getRequestId());
        return selectPinned(
                balanceContext,
                roleType,
                group,
                seqLen,
                selector,
                indicator,
                softQueuePlacement);
    }

    private SnapshotSelection selectFromSnapshots(
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            ResourceMeasureIndicatorEnum indicator,
            boolean softQueuePlacement) {
        EndpointFilterResult<EndpointRegistry.DecodeRoutingSnapshot> filterResult =
                getAvailableEndpointSnapshots(
                        group, indicator, softQueuePlacement);
        List<EndpointRegistry.DecodeRoutingSnapshot> eligible =
                filterResult.candidates();
        if (eligible.isEmpty()) {
            logNoAvailableEndpoint(balanceContext, filterResult);
            return SnapshotSelection.complete(null);
        }

        FilterResult<EndpointRegistry.DecodeRoutingSnapshot> hardFilterResult =
                applyHardFilters(
                        eligible,
                        SNAPSHOT_ROUTING,
                        seqLen,
                        selector,
                        softQueuePlacement);
        EndpointRegistry.DecodeRoutingSnapshot selected =
                weightedRandomSelection(
                        hardFilterResult,
                        SNAPSHOT_ROUTING,
                        selector.getDecayPerToken());
        if (selected == null) {
            logAllFilteredOut(
                    balanceContext,
                    filterResult.rejections(),
                    hardFilterResult.rejections());
            return SnapshotSelection.complete(null);
        }

        WorkerEndpoint.GenerationPin pin =
                engineWorkerStatus.captureCurrentDecodeWorker(
                        selected);
        if (pin == null) {
            return SnapshotSelection.conflict();
        }
        return SnapshotSelection.complete(buildSelectedRole(
                selected, pin, roleType, balanceContext));
    }

    private SelectedRole selectPinned(
            BalanceContext balanceContext,
            RoleType roleType,
            String group,
            long seqLen,
            RoutingConfig.KvUsageWeightedRandomConfig selector,
            ResourceMeasureIndicatorEnum indicator,
            boolean softQueuePlacement) {
        EndpointFilterResult<DecodeCandidate> filterResult = getAvailableEndpoints(
                roleType, group, indicator, softQueuePlacement);
        List<DecodeCandidate> eligible = filterResult.candidates();
        if (eligible.isEmpty()) {
            logNoAvailableEndpoint(balanceContext, filterResult);
            return null;
        }
        try {
            FilterResult<DecodeCandidate> hardFilterResult =
                    applyHardFilters(
                            eligible,
                            PINNED_ROUTING,
                            seqLen,
                            selector,
                            softQueuePlacement);
            DecodeCandidate selected = weightedRandomSelection(
                    hardFilterResult,
                    PINNED_ROUTING,
                    selector.getDecayPerToken());

            if (selected != null) {
                return buildSelectedRole(
                        selected, roleType, balanceContext);
            }

            logAllFilteredOut(
                    balanceContext,
                    filterResult.rejections(),
                    hardFilterResult.rejections());
            return null;
        } finally {
            for (DecodeCandidate candidate : eligible) {
                candidate.close();
            }
        }
    }

    @FunctionalInterface
    private interface RoutingCandidate<T> {
        DecodeEndpoint.DecodeRoutingView view(T candidate);
    }

    private static final RoutingCandidate<EndpointRegistry.DecodeRoutingSnapshot>
            SNAPSHOT_ROUTING = EndpointRegistry.DecodeRoutingSnapshot::routing;
    private static final RoutingCandidate<DecodeCandidate> PINNED_ROUTING =
            DecodeCandidate::view;

    private static final class DecodeCandidate
            implements AutoCloseable {
        private WorkerEndpoint.GenerationPin pin;
        private final DecodeEndpoint endpoint;
        private final DecodeEndpoint.DecodeRoutingView view;

        private DecodeCandidate(
                WorkerEndpoint.GenerationPin pin,
                DecodeEndpoint endpoint,
                DecodeEndpoint.DecodeRoutingView view) {
            this.pin = pin;
            this.endpoint = endpoint;
            this.view = view;
        }

        private DecodeEndpoint endpoint() {
            return endpoint;
        }

        public DecodeEndpoint.DecodeRoutingView view() {
            return view;
        }

        private WorkerEndpoint.GenerationPin requirePin() {
            WorkerEndpoint.GenerationPin owned = pin;
            if (owned == null) {
                throw new IllegalStateException(
                        "Decode candidate pin was already consumed");
            }
            return owned;
        }

        private WorkerEndpoint.GenerationPin takePin() {
            WorkerEndpoint.GenerationPin owned = requirePin();
            pin = null;
            return owned;
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
            ResourceMeasureIndicatorEnum indicator,
            boolean softQueuePlacement) {
        DecodeResourceMeasure measure =
                (DecodeResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult<>(
                    List.of(), Map.of("NO_REGISTERED", 1), 0);
        }
        List<EndpointRegistry.DecodeRoutingSnapshot> snapshots =
                engineWorkerStatus.decodeWorkerRoutingSnapshot(group);
        int registered = snapshots.size();
        if (registered == 0) {
            return new EndpointFilterResult<>(
                    snapshots, Map.of("NO_REGISTERED", 1), 0);
        }
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

    private EndpointFilterResult<DecodeCandidate> getAvailableEndpoints(
            RoleType roleType,
            String group,
            ResourceMeasureIndicatorEnum indicator,
            boolean softQueuePlacement) {
        DecodeResourceMeasure measure = (DecodeResourceMeasure) resourceMeasureFactory.getMeasure(indicator);
        if (measure == null) {
            return new EndpointFilterResult<>(
                    List.of(), Map.of("NO_REGISTERED", 1), 0);
        }
        List<WorkerEndpoint.GenerationPin> captured =
                engineWorkerStatus.captureModelWorkerEndpoints(roleType, group);
        List<DecodeCandidate> result = new ArrayList<>(captured.size());
        int unavailable = 0;
        try {
            for (int index = 0; index < captured.size(); index++) {
                WorkerEndpoint.GenerationPin pin = captured.get(index);
                if (!(pin.endpoint() instanceof DecodeEndpoint de)) {
                    pin.close();
                    captured.set(index, null);
                    continue;
                }
                DecodeEndpoint.DecodeRoutingView view = de.routingView();
                boolean available = softQueuePlacement
                        || measure.isResourceAvailable(view);
                if (!available) {
                    unavailable++;
                    pin.close();
                    captured.set(index, null);
                    continue;
                }
                result.add(new DecodeCandidate(pin, de, view));
                captured.set(index, null);
            }
        } catch (Throwable failure) {
            for (DecodeCandidate candidate : result) {
                candidate.close();
            }
            for (WorkerEndpoint.GenerationPin pin : captured) {
                if (pin != null) {
                    pin.close();
                }
            }
            throw failure;
        }
        int registered = captured.size();
        if (registered == 0) {
            return new EndpointFilterResult<>(
                    result, Map.of("NO_REGISTERED", 1), 0);
        }
        Map<String, Integer> rejections = unavailable == 0
                ? Map.of()
                : Map.of("RESOURCE_UNAVAILABLE", unavailable);
        return new EndpointFilterResult<>(result, rejections, registered);
    }

    private <T> FilterResult<T> applyHardFilters(
            List<T> eligible,
            RoutingCandidate<T> routing,
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
            DecodeEndpoint.DecodeRoutingView view = routing.view(candidate);
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
            DecodeEndpoint.DecodeRoutingView view = routing.view(candidate);
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
            RoutingCandidate<T> routing,
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
            long cacheUsed = routing.view(candidates.get(i)).realKvUsed();
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
            DecodeCandidate selected, RoleType roleType, BalanceContext balanceContext) {
        long requestId = balanceContext.getRequestId();
        ServerStatus result = new ServerStatus();
        DecodeEndpoint optimalEndpoint = selected.endpoint();
        WorkerStatus workerStatus = optimalEndpoint.getStatus();
        WorkerStatus.TopologySnapshot topology = workerStatus.topologySnapshot();
        DecodeEndpoint.DecodeRoutingView routing = selected.view();
        WorkerStatus.EngineObservation status = routing.workerStatus().fields();
        result.setSuccess(true);
        result.setRole(roleType);
        result.setServerIp(topology.ip());
        result.setHttpPort(topology.port());
        result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
        result.setDpRank(status.dpRank());
        result.setGroup(topology.group());
        result.setRequestId(requestId);
        return SelectedRole.decode(selected.takePin(), result, routing.totalKv());
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
