package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/** Random worker selection compatible with the current ownership transaction. */
@Component
public class RandomStrategy implements LoadBalanceStrategy {

    private static final org.slf4j.Logger LOGGER =
            LoggerFactory.getLogger(RandomStrategy.class);

    private final WorkerDirectory workerDirectory;
    private final ResourceMeasureFactory resourceMeasureFactory;

    public RandomStrategy(
            WorkerDirectory workerDirectory,
            ResourceMeasureFactory resourceMeasureFactory) {
        this.workerDirectory = workerDirectory;
        this.resourceMeasureFactory = resourceMeasureFactory;
    }

    @Override
    public boolean supports(
            RoleType role,
            RoutingConfig.EndpointSelectorConfig configured) {
        return switch (role) {
            case PREFILL, PDFUSION -> configured
                    instanceof RoutingConfig.RandomPrefillSelectorConfig;
            case DECODE -> configured
                    instanceof RoutingConfig.RandomDecodeSelectorConfig;
            case VIT -> configured
                    instanceof RoutingConfig.RandomVitSelectorConfig;
            case FRONTEND -> false;
        };
    }

    @Override
    public SelectedRole select(
            BalanceContext context,
            RoleType role,
            String group) {
        List<WorkerEndpoint.GenerationPin> candidates =
                workerDirectory.captureEndpoints(role, group);
        if (candidates.isEmpty()) {
            LOGGER.warn("No worker status map found for role {}", role);
            return null;
        }

        FlexlbConfig config = context.getConfig();
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(
                config.resourceMeasureFor(role));
        int start = ThreadLocalRandom.current().nextInt(candidates.size());
        try {
            for (int offset = 0; offset < candidates.size(); offset++) {
                int index = (start + offset) % candidates.size();
                WorkerEndpoint.GenerationPin pin = candidates.get(index);
                if (!isAvailable(pin.endpoint(), measure)) {
                    continue;
                }
                candidates.set(index, null);
                return buildSelection(pin, role, context, config);
            }
            LOGGER.warn(
                    "No serviceable {} worker available out of {} candidates",
                    role, candidates.size());
            return null;
        } finally {
            for (WorkerEndpoint.GenerationPin pin : candidates) {
                if (pin != null) {
                    pin.close();
                }
            }
        }
    }

    private static boolean isAvailable(
            WorkerEndpoint endpoint,
            ResourceMeasure measure) {
        if (endpoint instanceof DecodeEndpoint decode
                && measure instanceof DecodeResourceMeasure decodeMeasure) {
            return decodeMeasure.isResourceAvailable(decode.routingView());
        }
        if (endpoint instanceof PrefillEndpoint prefill
                && measure instanceof PrefillResourceMeasure prefillMeasure) {
            return prefillMeasure.isResourceAvailable(
                    prefill.admissionPendingRequestCount());
        }
        return true;
    }

    private static SelectedRole buildSelection(
            WorkerEndpoint.GenerationPin pin,
            RoleType role,
            BalanceContext context,
            FlexlbConfig config) {
        try {
            WorkerEndpoint endpoint = pin.endpoint();
            WorkerStatus.TopologySnapshot topology =
                    endpoint.getStatus().topologySnapshot();
            WorkerStatus.EngineObservation status =
                    endpoint.getStatus().committedEngineObservation();
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(role);
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setGroup(topology.group());
            result.setRequestId(context.getRequestId());

            if (role == RoleType.DECODE) {
                DecodeEndpoint decode = (DecodeEndpoint) endpoint;
                long totalKv = decode.routingView().totalKv();
                WorkerEndpoint.GenerationPin owned = pin;
                pin = null;
                return SelectedRole.decode(owned, result, totalKv);
            }
            if (endpoint instanceof PrefillEndpoint prefill) {
                long predictedMs = 0L;
                if (!config.isQueue()) {
                    predictedMs =
                            PrefillPredictionBoundary.predictSingleRequestMs(
                                    prefill.getPredictor().evaluator(),
                                    context.getRequest().getSeqLen(),
                                    0L);
                }
                WorkerEndpoint.GenerationPin owned = pin;
                pin = null;
                return SelectedRole.prefill(
                        owned, result, Math.max(0L, predictedMs));
            }
            WorkerEndpoint.GenerationPin owned = pin;
            pin = null;
            return SelectedRole.stateless(owned, result);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }
}
