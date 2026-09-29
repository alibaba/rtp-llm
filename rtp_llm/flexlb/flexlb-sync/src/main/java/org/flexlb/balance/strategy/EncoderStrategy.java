package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * Selects a healthy Encoder using observed requests and uncached MM work.
 */
@Component
public final class EncoderStrategy {

    private final WorkerDirectory workerDirectory;

    /**
     * Select from the current worker directory and its pinned endpoint generations.
     */
    public EncoderStrategy(WorkerDirectory workerDirectory) {
        this.workerDirectory = Objects.requireNonNull(workerDirectory, "workerDirectory");
    }

    /**
     * Choose by uncached MM work when the client supplies a cache estimate.
     * Older requests continue to choose by running, waiting, and locally pending count.
     * Returns null when no worker qualifies.
     */
    public SelectedRole select(BalanceContext context, String group) {
        context.beginRoutingAttempt(RoleType.ENCODER);
        int maxInflightPerWorker = context.getConfig().isQueue()
                ? context.getConfig().getDispatcher().getMaxInflightPerEncoderWorker()
                : Integer.MAX_VALUE;
        WorkerEndpoint.GenerationPin winner = null;
        WorkerStatus.TopologySnapshot winnerTopology = null;
        WorkerStatus.EngineObservation winnerEngine = null;
        boolean weighted = context.getRequest().getEncoderCacheHitLen() != null;
        long leastUncachedTokens = Long.MAX_VALUE;
        long fewestRequests = Long.MAX_VALUE;
        try {
            for (String address : workerDirectory.endpointAddressSnapshot(RoleType.ENCODER)) {
                WorkerEndpoint.GenerationPin candidate = workerDirectory.captureEndpoint(RoleType.ENCODER, address);
                if (candidate == null) {
                    continue;
                }
                try {
                    EncoderEndpoint endpoint = (EncoderEndpoint) candidate.endpoint();
                    WorkerStatus status = endpoint.getStatus();
                    WorkerStatus.TopologySnapshot topology = status.topologySnapshot();
                    WorkerStatus.EngineObservation engine = status.committedEngineObservation();
                    if (!status.isAlive()
                            || group != null && !group.equals(topology.group())) {
                        continue;
                    }
                    long requests = Math.max(0, engine.runningQueryLen())
                            + Math.max(0, engine.waitingQueryLen())
                            + endpoint.pendingEncoderRequestCount();
                    if (requests >= maxInflightPerWorker) {
                        continue;
                    }
                    long uncachedTokens = weighted ? endpoint.inflightUncachedTokenEstimate() : 0L;
                    if (uncachedTokens > leastUncachedTokens
                            || uncachedTokens == leastUncachedTokens && requests >= fewestRequests) {
                        continue;
                    }
                    if (winner != null) {
                        winner.close();
                    }
                    winner = candidate;
                    candidate = null;
                    winnerTopology = topology;
                    winnerEngine = engine;
                    leastUncachedTokens = uncachedTokens;
                    fewestRequests = requests;
                } finally {
                    if (candidate != null) {
                        candidate.close();
                    }
                }
            }
            if (winner == null) {
                return null;
            }
            ServerStatus result = new ServerStatus();
            result.setSuccess(true);
            result.setRole(RoleType.ENCODER);
            result.setRequestId(context.getRequestId());
            result.setGroup(winnerTopology.group());
            result.setServerIp(winnerTopology.ip());
            result.setHttpPort(winnerTopology.port());
            result.setGrpcPort(CommonUtils.toGrpcPort(winnerTopology.port()));
            result.setDpRank(winnerEngine.dpRank());
            result.setSelectedEngineIndex(winnerTopology.engineIndex(), winnerTopology.multiEngineNum());
            context.recordSelectionReason(RoleType.ENCODER,
                    weighted ? "LEAST_UNCACHED_TOKENS_CONCURRENCY" : "LEAST_CONCURRENT");
            SelectedRole selected = SelectedRole.stateless(winner, result);
            winner = null;
            return selected;
        } finally {
            if (winner != null) {
                winner.close();
            }
        }
    }
}
