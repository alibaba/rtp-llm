package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
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

@Component
public class RandomStrategy implements LoadBalanceStrategy {
    private static final org.slf4j.Logger logger = LoggerFactory.getLogger(RandomStrategy.class);

    private final WorkerDirectory workerDirectory;
    private final PrefillResourceMeasure prefillResourceMeasure;
    private final DecodeResourceMeasure decodeResourceMeasure;

    public RandomStrategy(WorkerDirectory workerDirectory,
                          PrefillResourceMeasure prefillResourceMeasure,
                          DecodeResourceMeasure decodeResourceMeasure) {
        this.workerDirectory = workerDirectory;
        this.prefillResourceMeasure = prefillResourceMeasure;
        this.decodeResourceMeasure = decodeResourceMeasure;
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
    public SelectedRole select(BalanceContext balanceContext, RoleType roleType, String group) {
        logger.debug("Selecting worker for , role: {}, group: {}", roleType, group);

        FlexlbConfig config = balanceContext.getConfig();

        List<String> candidateAddresses =
                workerDirectory.endpointAddressSnapshot(roleType);
        if (candidateAddresses.isEmpty()) {
            logger.warn("No worker status map found");
            return null;
        }

        // Random start with wrap-around preserves the existing fast-path
        // selection order without pinning every endpoint generation up front.
        // Each address is re-entered through EndpointRegistry.capture, so a
        // detach or same-address replacement still linearizes exactly.
        int size = candidateAddresses.size();
        int startIndex = ThreadLocalRandom.current().nextInt(size);
        PhysicalDecodeCapacity physicalCapacity = roleType == RoleType.DECODE
                ? new PhysicalDecodeCapacity(
                        balanceContext.getRequest().getSeqLen())
                : null;
        boolean softDecodePlacement = roleType == RoleType.DECODE
                && config.defersDecodeCapacityUntilDispatch();
        int selectionPasses = softDecodePlacement ? 2 : 1;
        for (int pass = 0; pass < selectionPasses; pass++) {
            boolean allowTransientlyFull = softDecodePlacement && pass == 1;
            for (int offset = 0; offset < size; offset++) {
                String address = candidateAddresses.get((startIndex + offset) % size);
                WorkerEndpoint.GenerationPin pin =
                        workerDirectory.captureEndpoint(roleType, address);
                if (pin == null) {
                    if (physicalCapacity != null) {
                        physicalCapacity.markUnknown();
                    }
                    continue;
                }
                try {
                    RoutingCandidate selected = snapshotIfAvailable(
                            config, roleType, group, pin, physicalCapacity,
                            allowTransientlyFull);
                    if (selected == null) {
                        continue;
                    }

                    WorkerEndpoint selectedWorker = selected.endpoint();
                    logger.debug("Selected worker ip: {}, httpPort: {}",
                            selectedWorker.getIp(), selectedWorker.getHttpPort());

                    // buildSelectedRole takes ownership immediately and either
                    // returns a SelectedRole or closes the exact pin on failure.
                    WorkerEndpoint.GenerationPin selectedPin = pin;
                    pin = null;
                    return buildSelectedRole(
                            selectedPin, selected,
                            roleType, balanceContext, config);
                } finally {
                    if (pin != null) {
                        pin.close();
                    }
                }
            }
        }
        if (physicalCapacity != null) {
            physicalCapacity.rejectIfImpossible();
        }
        logger.warn("No serviceable workers available out of {} total workers", size);
        return null;
    }

    private record RoutingCandidate(
            WorkerEndpoint endpoint,
            DecodeEndpoint.DecodeRoutingView decodeView,
            WorkerStatus.TopologySnapshot topology) {}

    private RoutingCandidate snapshotIfAvailable(
            FlexlbConfig config,
            RoleType roleType,
            String group,
            WorkerEndpoint.GenerationPin pin,
            PhysicalDecodeCapacity physicalCapacity,
            boolean allowTransientlyFull) {
        WorkerEndpoint ep = pin.endpoint();
        if (ep == null) {
            return null;
        }
        WorkerStatus.TopologySnapshot topology =
                ep.getStatus().topologySnapshot();
        if (group != null && !group.equals(topology.group())) {
            return null;
        }

        if (ep instanceof DecodeEndpoint decodeEndpoint) {
            DecodeEndpoint.DecodeRoutingView view = decodeEndpoint.routingView();
            if (!physicalCapacity.accepts(view.totalKv())) {
                return null;
            }
            boolean available = allowTransientlyFull
                    || (config.defersDecodeCapacityUntilDispatch()
                            ? decodeResourceMeasure.isEngineDispatchAvailable(view)
                            : decodeResourceMeasure.isResourceAvailable(view));
            return available ? new RoutingCandidate(ep, view, topology) : null;
        }
        if (ep instanceof PrefillEndpoint prefillEndpoint
                && !prefillResourceMeasure.isResourceAvailable(
                        prefillEndpoint.admissionPendingRequestCount())) {
            return null;
        }
        return new RoutingCandidate(ep, null, topology);
    }

    /** Tracks immutable capacity without treating a transient load as static. */
    private static final class PhysicalDecodeCapacity {
        private final long requiredKv;
        private long maximumKnown;
        private boolean observed;
        private boolean unknown;

        private PhysicalDecodeCapacity(long requiredKv) {
            this.requiredKv = Math.max(0L, requiredKv);
        }

        private boolean accepts(long physicalKv) {
            observed = true;
            if (physicalKv <= 0L) {
                unknown = true;
                return true;
            }
            maximumKnown = Math.max(maximumKnown, physicalKv);
            return requiredKv <= physicalKv;
        }

        private void markUnknown() {
            unknown = true;
        }

        private void rejectIfImpossible() {
            if (observed && !unknown && requiredKv > maximumKnown) {
                throw new StaticCapacityExceededException(
                        "Decode request seq_len=" + requiredKv
                                + " exceeds max known physical KV="
                                + maximumKnown);
            }
        }
    }

    private SelectedRole buildSelectedRole(
            WorkerEndpoint.GenerationPin selectedPin,
            RoutingCandidate selected,
            RoleType roleType,
            BalanceContext balanceContext,
            FlexlbConfig config) {
        try {
            long requestId = balanceContext.getRequestId();
            ServerStatus result = new ServerStatus();
            WorkerEndpoint ep = selected.endpoint();
            if (selectedPin.endpoint() != ep) {
                throw new IllegalStateException(
                        "random candidate pin identity changed before selection");
            }
            WorkerStatus workerStatus = ep.getStatus();
            WorkerStatus.TopologySnapshot topology = selected.topology();
            DecodeEndpoint decodeEndpoint =
                    roleType == RoleType.DECODE && ep instanceof DecodeEndpoint decode
                            ? decode : null;
            DecodeEndpoint.DecodeRoutingView decodeView =
                    decodeEndpoint == null ? null : selected.decodeView();
            WorkerStatus.EngineObservation status =
                    decodeView == null
                            ? workerStatus.committedEngineObservation()
                            : decodeView.workerStatus().fields();
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(status.dpRank());
            result.setRole(roleType);
            result.setGroup(topology.group());
            result.setRequestId(requestId);
            if (roleType == RoleType.DECODE && decodeEndpoint == null) {
                throw new IllegalStateException(
                        "DECODE random selection requires DecodeEndpoint ownership");
            }
            long predictedMs = -1L;
            if (roleType == RoleType.PREFILL && !config.isQueue()) {
                if (!(ep instanceof PrefillEndpoint prefillEndpoint)) {
                    throw new IllegalStateException(
                            "PREFILL random selection requires PrefillEndpoint ownership");
                }
                predictedMs =
                        PrefillPredictionBoundary.predictSingleRequestMs(
                                prefillEndpoint.getPredictor().evaluator(),
                                balanceContext.getRequest().getSeqLen(),
                                0L);
            }
            result.setSuccess(true);

            // Resolve every fallible value before the SelectedRole factory.
            // The factory consumes the exact pin, including validation failure.
            if (roleType == RoleType.DECODE) {
                long decodeTotalKv = selected.decodeView().totalKv();
                WorkerEndpoint.GenerationPin factoryPin = selectedPin;
                selectedPin = null;
                return SelectedRole.decode(factoryPin, result, decodeTotalKv);
            }
            if (ep instanceof PrefillEndpoint) {
                long prefillWorkMs = Math.max(0L, predictedMs);
                WorkerEndpoint.GenerationPin factoryPin = selectedPin;
                selectedPin = null;
                // QUEUE prediction is performed later by the admission
                // snapshot; only DIRECT needs a route-time work estimate.
                return SelectedRole.prefill(factoryPin, result, prefillWorkMs);
            }
            WorkerEndpoint.GenerationPin factoryPin = selectedPin;
            selectedPin = null;
            return SelectedRole.stateless(factoryPin, result);
        } finally {
            if (selectedPin != null) {
                selectedPin.close();
            }
        }
    }
}
