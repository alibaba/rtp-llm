package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Frozen facts for one selected role. Endpoint resources remain live; their selection-time
 * version and requested work belong to the role that uses them, never to a resource ledger.
 */
public abstract sealed class RoleSelection {
    private final WorkerEndpoint endpoint;
    private final RoleType role;
    private final WorkerStatus.TopologySnapshot topology;
    private final long dpRank;
    private final long workerGeneration;
    private final long prefillTime;
    private final DebugInfo debugInfo;
    private final List<String> preemptRequestIds;

    private RoleSelection(WorkerEndpoint endpoint, ServerStatus status) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        role = status.getRole();
        topology =
                new WorkerStatus.TopologySnapshot(
                        status.getGroup(),
                        status.getServerIp(),
                        status.getHttpPort(),
                        status.getGrpcPort(),
                        null,
                        null,
                        status.getRoutingEngineIndex(),
                        status.getRoutingMultiEngineNum());
        dpRank = status.getDpRank();
        workerGeneration = status.getWorkerGeneration();
        prefillTime = status.getPrefillTime();
        debugInfo = DebugInfo.copyOf(status.getDebugInfo());
        preemptRequestIds = List.copyOf(status.getPreemptRequestIds());
    }

    static RoleSelection capture(
            WorkerEndpoint endpoint, ServerStatus status, long workMs, long version) {
        checkArgument(version >= 0L, "placementVersion must be non-negative");
        return switch (status.getRole()) {
            case PREFILL, PDFUSION -> {
                checkArgument(endpoint instanceof PrefillEndpoint, "Prefill endpoint required");
                checkArgument(workMs >= 0L, "Prefill work must be non-negative");
                yield new Prefill(endpoint, status, workMs, version);
            }
            case DECODE -> {
                checkArgument(endpoint instanceof DecodeEndpoint, "Decode endpoint required");
                yield new Decode(endpoint, status, version);
            }
            case ENCODER -> {
                checkArgument(endpoint instanceof EncoderEndpoint, "Encoder endpoint required");
                yield new Auxiliary(endpoint, status);
            }
            case VIT -> new Auxiliary(endpoint, status);
            case FRONTEND -> throw new IllegalArgumentException("unsupported Worker role");
        };
    }

    public final WorkerEndpoint endpoint() {
        return endpoint;
    }

    public final RoleType role() {
        return role;
    }

    public final long dpRank() {
        return dpRank;
    }

    public final long hitCacheTokens() {
        return debugInfo == null ? 0L : debugInfo.getHitCacheLen();
    }

    public final PlacementKey placementKey() {
        return PlacementKey.exact(
                role,
                topology.group(),
                new org.flexlb.dao.master.WorkerIdentity(
                                topology.ip(), topology.port(), topology.engineIndex())
                        .getLogicalIpPort());
    }

    /**
     * Builds a fresh protocol projection; callers cannot mutate the selected facts.
     */
    public final ServerStatus toServerStatus(String requestId) {
        var status = new ServerStatus();
        status.setSuccess(true);
        status.setRequestId(requestId);
        status.setRole(role);
        status.setGroup(topology.group());
        status.setServerIp(topology.ip());
        status.setHttpPort(topology.port());
        status.setGrpcPort(topology.grpcPort());
        status.setSelectedEngineIndex(topology.engineIndex(), topology.multiEngineNum());
        status.setDpRank(dpRank);
        status.setWorkerGeneration(workerGeneration);
        status.setPrefillTime(prefillTime);
        status.setDebugInfo(DebugInfo.copyOf(debugInfo));
        status.setPreemptRequestIds(preemptRequestIds);
        return status;
    }

    public static final class Prefill extends RoleSelection {
        private final long workMs;
        private final long resourceVersion;

        private Prefill(
                WorkerEndpoint endpoint, ServerStatus status, long workMs, long resourceVersion) {
            super(endpoint, status);
            this.workMs = workMs;
            this.resourceVersion = resourceVersion;
        }

        public long workMs() {
            return workMs;
        }

        public long resourceVersion() {
            return resourceVersion;
        }
    }

    public static final class Decode extends RoleSelection {
        private final long resourceVersion;

        private Decode(WorkerEndpoint endpoint, ServerStatus status, long resourceVersion) {
            super(endpoint, status);
            this.resourceVersion = resourceVersion;
        }

        public long resourceVersion() {
            return resourceVersion;
        }
    }

    /**
     * Encoder and VIT currently need no selection-time resource estimates.
     */
    public static final class Auxiliary extends RoleSelection {
        private Auxiliary(WorkerEndpoint endpoint, ServerStatus status) {
            super(endpoint, status);
        }
    }
}
