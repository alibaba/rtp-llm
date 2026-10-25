package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/** Random selection for the stateless VIT role. */
@Component
public final class RandomStrategy {
    private static final Logger LOGGER =
            LoggerFactory.getLogger(RandomStrategy.class);

    private final WorkerDirectory workerDirectory;

    public RandomStrategy(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    public SelectedRole select(BalanceContext context, RoleType role, String group) {
        context.beginRoutingAttempt(role);
        if (role != RoleType.VIT) {
            throw new IllegalArgumentException(
                    "RANDOM endpoint selection is supported only for VIT");
        }
        List<String> addresses =
                workerDirectory.endpointAddressSnapshot(RoleType.VIT);
        if (addresses.isEmpty()) {
            return null;
        }

        int start = ThreadLocalRandom.current().nextInt(addresses.size());
        for (int offset = 0; offset < addresses.size(); offset++) {
            String address = addresses.get((start + offset) % addresses.size());
            WorkerEndpoint.EndpointHold hold = workerDirectory.holdEndpoint(RoleType.VIT, address);
            if (hold == null) {
                continue;
            }
            try {
                WorkerStatus status = hold.endpoint().getStatus();
                WorkerStatus.TopologySnapshot topology =
                        status.topologySnapshot();
                if (group != null && !group.equals(topology.group())) {
                    continue;
                }
                WorkerStatus.EngineObservation engine =
                        status.committedEngineObservation();
                context.recordSelectionReason(role, "RANDOM");
                WorkerEndpoint.EndpointHold selectedHold = hold;
                hold = null;
                return selected(selectedHold, topology, engine, context.getRequestId());
            } finally {
                if (hold != null) {
                    hold.close();
                }
            }
        }
        LOGGER.warn(
                "No VIT worker available out of {} registered workers",
                addresses.size());
        return null;
    }

    private static SelectedRole selected(WorkerEndpoint.EndpointHold hold,
                                         WorkerStatus.TopologySnapshot topology,
                                         WorkerStatus.EngineObservation engine,
                                         String requestId) {
        try {
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(RoleType.VIT);
            result.setRequestId(requestId);
            result.setGroup(topology.group());
            result.setServerIp(topology.ip());
            result.setHttpPort(topology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
            result.setDpRank(engine.dpRank());
            result.setSelectedEngineIndex(
                    topology.engineIndex(), topology.multiEngineNum());

            WorkerEndpoint.EndpointHold owned = hold;
            hold = null;
            return SelectedRole.stateless(owned, result);
        } finally {
            if (hold != null) {
                hold.close();
            }
        }
    }
}
