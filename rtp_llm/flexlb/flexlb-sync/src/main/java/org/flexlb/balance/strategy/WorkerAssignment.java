package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Failures;

import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * A request's role assignment to one exact Worker generation, with frozen routing facts.
 * Its pin protects acquisition while the generation retires. DTO getters return copies;
 * pin lifetime is maintained solely by the underlying generation permit.</p>
 */
public final class WorkerAssignment implements AutoCloseable {

    private final WorkerEndpoint.GenerationPin generationPin;
    private final AtomicBoolean assigned = new AtomicBoolean();
    private final ServerStatus serverStatus;
    private final long prefillWorkMs;
    private final long placementVersion;

    private WorkerAssignment(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion) {
        this.generationPin = generationPin;
        this.serverStatus = ServerStatus.copyOf(serverStatus);
        checkArgument(serverStatus.isSuccess(), "WorkerAssignment requires successful response metadata");
        WorkerEndpoint endpoint = generationPin.endpoint();
        checkArgument(Objects.equals(serverStatus.getServerIp(), endpoint.getIp())
                && serverStatus.getHttpPort() == endpoint.getHttpPort(),
                "selection metadata does not match pinned endpoint address");
        checkArgument(prefillWorkMs < 0L
                || (endpoint instanceof PrefillEndpoint)
                && (serverStatus.getRole() == RoleType.PREFILL
                || serverStatus.getRole() == RoleType.PDFUSION),
                "Prefill selection requires a Prefill endpoint role");
        checkArgument(serverStatus.getRole() != RoleType.DECODE || (endpoint instanceof DecodeEndpoint),
                "Decode selection requires a Decode endpoint role");
        this.prefillWorkMs = prefillWorkMs;
        checkArgument(placementVersion >= 0L, "placementVersion must be non-negative");
        this.placementVersion = placementVersion;
    }

    public static WorkerAssignment prefill(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion) {
        if (prefillWorkMs < 0L) {
            try (generationPin) {
                throw new IllegalArgumentException("Prefill work must be non-negative");
            }
        }
        return createOwned(
                generationPin, serverStatus, prefillWorkMs,
                placementVersion);
    }

    public static WorkerAssignment decode(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long placementVersion) {
        if (serverStatus == null || serverStatus.getRole() != RoleType.DECODE) {
            try (generationPin) {
                throw new IllegalArgumentException("Decode selection requires Decode metadata");
            }
        }
        return createOwned(generationPin, serverStatus, -1L, placementVersion);
    }

    public static WorkerAssignment stateless(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus) {
        return createOwned(
                generationPin, serverStatus, -1L, 0L);
    }

    /** Calling a factory consumes the pin, including every validation failure. */
    private static WorkerAssignment createOwned(
            WorkerEndpoint.GenerationPin generationPin,
            ServerStatus serverStatus,
            long prefillWorkMs,
            long placementVersion) {
        Throwable failure = null;
        try {
            return new WorkerAssignment(generationPin, serverStatus, prefillWorkMs, placementVersion);
        } catch (RuntimeException | Error constructionFailure) {
            failure = constructionFailure;
            throw constructionFailure;
        } finally {
            if (failure != null && generationPin != null) {
                Failures.append(failure, Failures.close(generationPin));
            }
        }
    }

    static ServerStatus workerMetadata(RoleType role, long requestId,
            WorkerStatus.TopologySnapshot topology, WorkerStatus.EngineObservation observation) {
        ServerStatus result = new ServerStatus();
        result.setSuccess(true);
        result.setRole(role);
        result.setRequestId(requestId);
        result.setGroup(topology.group());
        result.setServerIp(topology.ip());
        result.setHttpPort(topology.port());
        result.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
        result.setDpRank(observation.dpRank());
        return result;
    }

    public ServerStatus serverStatus() {
        return ServerStatus.copyOf(serverStatus);
    }

    public long prefillWorkMs() {
        checkState(prefillWorkMs >= 0L, "selection does not carry Prefill work");
        return prefillWorkMs;
    }

    public long placementVersion() {
        return placementVersion;
    }

    /** One request assignment may consume this pinned Worker capability. */
    public void assignToRequest() {
        checkState(assigned.compareAndSet(false, true) && generationPin.isOpen(),
                "Worker generation was already assigned or closed");
    }

    public long requestId() { return serverStatus.getRequestId(); }
    public RoleType role() { return serverStatus.getRole(); }
    public String group() { return serverStatus.getGroup(); }
    public long hitCache() {
        return serverStatus.getDebugInfo() == null ? 0L : serverStatus.getDebugInfo().getHitCacheLen();
    }

    public WorkerEndpoint endpoint() {
        return generationPin.endpoint();
    }

    public WorkerEndpoint.GenerationPin generationPin() {
        checkState(generationPin.isOpen(), "assigned endpoint generation is closed");
        return generationPin;
    }

    @Override
    public void close() {
        generationPin.close();
    }
}
