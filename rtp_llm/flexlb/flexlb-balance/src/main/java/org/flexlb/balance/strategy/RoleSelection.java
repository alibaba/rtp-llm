package org.flexlb.balance.strategy;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.DebugInfo;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus.TopologySnapshot;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Retains selected topology, request-specific scores and the exact resource registration.
 * Role and rank come from the Endpoint; response projection never changes the saved selection.
 */
public final class RoleSelection {
    private final WorkerEndpoint endpoint;
    private final TopologySnapshot topology;

    private final long hitCacheTokens;
    private final long computeTokens;

    private final WorkerEndpoint.RequestRegistration registration;

    private RoleSelection(
            WorkerEndpoint endpoint,
            TopologySnapshot topology,
            long hitCacheTokens,
            long computeTokens,
            WorkerEndpoint.RequestRegistration registration) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        this.topology = Objects.requireNonNull(topology, "topology");
        checkArgument(
                topology.multiEngineNum() >= 1
                        && topology.engineIndex() >= 0
                        && topology.engineIndex() < topology.multiEngineNum(),
                "selected engine index must be in [0, multiEngineNum)");
        this.hitCacheTokens = hitCacheTokens;
        this.computeTokens = computeTokens;
        this.registration = registration;
    }

    public WorkerEndpoint endpoint() {
        return endpoint;
    }

    /**
     * Builds a fresh protocol projection; callers cannot mutate the selected facts.
     */
    public ServerStatus toServerStatus(String requestId) {
        var role = role();
        var response = new ServerStatus();
        response.setSuccess(true);
        response.setRole(role);
        response.setRequestId(requestId);
        response.setGroup(topology.group());
        response.setServerIp(topology.ip());
        response.setHttpPort(topology.port());
        response.setGrpcPort(topology.grpcPort());
        response.setSelectedEngineIndex(topology.engineIndex(), topology.multiEngineNum());
        response.setDpRank(role == RoleType.VIT ? 0L : endpoint.getStatus().getDpRank());
        if (role == RoleType.VIT)
            response.setWorkerGeneration(endpoint.getStatus().getGenerationId());
        if (role.supportsPrefill()) {
            var debug = new DebugInfo();
            debug.setHitCacheLen(hitCacheTokens);
            response.setDebugInfo(debug);
        }
        return response;
    }

    public RoleType role() {
        return endpoint.getStatus().getRole();
    }

    public static RoleSelection prefill(
            PrefillEndpoint endpoint,
            TopologySnapshot topology,
            long hitCacheTokens,
            long computeTokens,
            PrefillEndpoint.RequestRegistration registration) {
        checkArgument(computeTokens >= 0L, "Prefill work must be non-negative");
        return new RoleSelection(endpoint, topology, hitCacheTokens, computeTokens, registration);
    }

    public static RoleSelection decode(
            DecodeEndpoint endpoint,
            TopologySnapshot topology,
            DecodeEndpoint.RequestRegistration registration) {
        return new RoleSelection(endpoint, topology, 0L, 0L, registration);
    }

    public static RoleSelection encoder(
            EncoderEndpoint endpoint,
            TopologySnapshot topology,
            EncoderEndpoint.RequestRegistration registration) {
        return new RoleSelection(endpoint, topology, 0L, 0L, registration);
    }

    public static RoleSelection auxiliary(WorkerEndpoint endpoint, TopologySnapshot topology) {
        return new RoleSelection(endpoint, topology, 0L, 0L, null);
    }

    public WorkerEndpoint.RequestRegistration registration() {
        return registration;
    }

    public long computeTokens() {
        return computeTokens;
    }

    public CapacityKey capacityKey() {
        return CapacityKey.exact(
                role(),
                group(),
                topology.ip()
                        + ":"
                        + topology.port()
                        + org.flexlb.util.CommonConstants.LOGICAL_WORKER_ENGINE_INDEX_SEPARATOR
                        + topology.engineIndex());
    }

    String group() {
        return topology.group();
    }
}
