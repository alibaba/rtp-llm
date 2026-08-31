package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Decode selection policy from feat/dsv4_on_dev.
 *
 * <p>The policy is intentionally small: filter unavailable workers, reject
 * hard KV/load outliers, then choose by KV-usage weighted random. Endpoint
 * generation pins and reservation ownership remain in the current router and
 * admission transaction.
 */
@Component
public class CostBasedDecodeStrategy implements LoadBalanceStrategy {

    private final WorkerDirectory workerDirectory;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public CostBasedDecodeStrategy(
            WorkerDirectory workerDirectory,
            ResourceMeasureFactory resourceMeasureFactory) {
        this.workerDirectory = workerDirectory;
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
    public SelectedRole select(
            BalanceContext context,
            RoleType role,
            String group) {
        FlexlbConfig config = context.getConfig();
        RoutingConfig.KvUsageWeightedRandomConfig selector =
                (RoutingConfig.KvUsageWeightedRandomConfig) config.getRouter()
                        .getRoles().getDecode().getSelector();
        DecodeResourceMeasure measure = (DecodeResourceMeasure)
                resourceMeasureFactory.getMeasure(
                        config.resourceMeasureFor(role));

        List<Candidate> candidates = captureCandidates(
                role, group, measure, context.getRequestId());
        try {
            List<Candidate> survivors = applyHardFilters(
                    candidates, context.getRequest().getSeqLen(), selector);
            Candidate selected = weightedRandomSelection(
                    survivors, selector.getDecayPerToken());
            if (selected == null) {
                Logger.debug(
                        "Decode select failed: no available worker, request_id={}",
                        context.getRequestId());
                return null;
            }
            return selected.takeSelection(role, context);
        } finally {
            closeCandidates(candidates);
        }
    }

    private List<Candidate> captureCandidates(
            RoleType role,
            String group,
            DecodeResourceMeasure measure,
            long requestId) {
        if (measure == null) {
            return List.of();
        }
        List<WorkerEndpoint.GenerationPin> pins =
                workerDirectory.captureEndpoints(role, group);
        List<Candidate> candidates = new ArrayList<>(pins.size());
        try {
            for (int index = 0; index < pins.size(); index++) {
                WorkerEndpoint.GenerationPin pin = pins.get(index);
                WorkerEndpoint endpoint = pin.endpoint();
                if (!(endpoint instanceof DecodeEndpoint decode)) {
                    pin.close();
                    pins.set(index, null);
                    continue;
                }
                DecodeEndpoint.DecodeRoutingView view = decode.routingView();
                if (!measure.isResourceAvailable(view)) {
                    pin.close();
                    pins.set(index, null);
                    continue;
                }
                candidates.add(new Candidate(pin, decode, view));
                pins.set(index, null);
            }
            if (candidates.isEmpty()) {
                Logger.debug(
                        "Decode select failed: no resource-available endpoint,"
                                + " request_id={}",
                        requestId);
            }
            return candidates;
        } catch (Throwable failure) {
            closeCandidates(candidates);
            throw failure;
        } finally {
            for (WorkerEndpoint.GenerationPin pin : pins) {
                if (pin != null) {
                    pin.close();
                }
            }
        }
    }

    private static List<Candidate> applyHardFilters(
            List<Candidate> eligible,
            long sequenceLength,
            RoutingConfig.KvUsageWeightedRandomConfig selector) {
        if (eligible.isEmpty()) {
            return List.of();
        }
        RoutingConfig.DecodeOutlierRejectionConfig outlier =
                selector.getOutlierRejection();
        double hotspotMultiplier = outlier == null
                ? 0.0
                : outlier.getMaxEngineLoadVsAverageMultiplier();
        double imbalanceMultiplier = outlier == null
                ? 0.0
                : outlier.getMaxKvUsedVsAverageMultiplier();

        long sumLoad = 0L;
        long sumKvUsed = 0L;
        for (Candidate candidate : eligible) {
            sumLoad += candidate.view.engineLoad();
            sumKvUsed += candidate.view.realKvUsed();
        }
        long averageLoad = sumLoad / eligible.size();
        long averageKvUsed = sumKvUsed / eligible.size();

        List<Candidate> survivors = new ArrayList<>(eligible.size());
        Map<String, Integer> rejections = new java.util.HashMap<>();
        for (Candidate candidate : eligible) {
            DecodeEndpoint.DecodeRoutingView view = candidate.view;
            if (view.totalKv() > 0L
                    && view.realKvAvailable() < sequenceLength) {
                rejections.merge("KV_CAPACITY", 1, Integer::sum);
                continue;
            }
            if (hotspotMultiplier > 0.0
                    && averageLoad > 0L
                    && view.engineLoad()
                            > averageLoad * hotspotMultiplier) {
                rejections.merge("HOTSPOT_FILTERED", 1, Integer::sum);
                continue;
            }
            if (imbalanceMultiplier > 0.0
                    && averageKvUsed > 0L
                    && view.realKvUsed()
                            > averageKvUsed * imbalanceMultiplier) {
                rejections.merge("IMBALANCE_FILTERED", 1, Integer::sum);
                continue;
            }
            survivors.add(candidate);
        }
        if (survivors.isEmpty() && !rejections.isEmpty()) {
            Logger.debug(
                    "Decode select failed: all candidates filtered, rejections={}",
                    rejections);
        }
        return survivors;
    }

    private static Candidate weightedRandomSelection(
            List<Candidate> candidates,
            double decayPerToken) {
        if (candidates.isEmpty()) {
            return null;
        }

        int minimumIndex = 0;
        int maximumIndex = 0;
        boolean allSameUsage = true;
        long firstUsage = candidates.getFirst().view.realKvUsed();
        for (int index = 1; index < candidates.size(); index++) {
            long usage = candidates.get(index).view.realKvUsed();
            if (usage != firstUsage) {
                allSameUsage = false;
            }
            if (usage < candidates.get(minimumIndex).view.realKvUsed()) {
                minimumIndex = index;
            }
            if (usage > candidates.get(maximumIndex).view.realKvUsed()) {
                maximumIndex = index;
            }
        }
        if (allSameUsage) {
            return candidates.get(
                    ThreadLocalRandom.current().nextInt(candidates.size()));
        }

        long referenceUsage = decayPerToken >= 0.0
                ? candidates.get(minimumIndex).view.realKvUsed()
                : candidates.get(maximumIndex).view.realKvUsed();
        double[] weights = new double[candidates.size()];
        double totalWeight = 0.0;
        for (int index = 0; index < candidates.size(); index++) {
            double normalized = (double)
                    (candidates.get(index).view.realKvUsed()
                            - referenceUsage);
            weights[index] = Math.exp(-decayPerToken * normalized);
            totalWeight += weights[index];
        }
        if (!Double.isFinite(totalWeight) || totalWeight <= 0.0) {
            return candidates.get(minimumIndex);
        }

        double draw = ThreadLocalRandom.current().nextDouble(totalWeight);
        double cumulative = 0.0;
        for (int index = 0; index < candidates.size(); index++) {
            cumulative += weights[index];
            if (draw <= cumulative) {
                return candidates.get(index);
            }
        }
        return candidates.get(minimumIndex);
    }

    private static void closeCandidates(List<Candidate> candidates) {
        for (Candidate candidate : candidates) {
            candidate.close();
        }
    }

    private static final class Candidate implements AutoCloseable {
        private WorkerEndpoint.GenerationPin pin;
        private final DecodeEndpoint endpoint;
        private final DecodeEndpoint.DecodeRoutingView view;

        private Candidate(
                WorkerEndpoint.GenerationPin pin,
                DecodeEndpoint endpoint,
                DecodeEndpoint.DecodeRoutingView view) {
            this.pin = pin;
            this.endpoint = endpoint;
            this.view = view;
        }

        private SelectedRole takeSelection(
                RoleType role,
                BalanceContext context) {
            WorkerEndpoint.GenerationPin owned = pin;
            if (owned == null) {
                throw new IllegalStateException(
                        "Decode candidate pin already consumed");
            }
            WorkerStatus.TopologySnapshot topology =
                    endpoint.getStatus().topologySnapshot();
            WorkerStatus.EngineObservation status =
                    view.workerStatus().fields();
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(role);
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setGroup(topology.group());
            result.setRequestId(context.getRequestId());
            pin = null;
            return SelectedRole.decode(
                    owned, result, view.totalKv());
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
