package org.flexlb.balance.strategy;

import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.SelectionRollback;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * One selected role and its exact registration. Metadata is frozen; the DiscoveryEndpoint owns live capacity.
 * The same registration is retained from selection through queueing and delivery.
 */
public final class RoleSelection {
    private final WorkerEndpoint endpoint;
    private final ServerStatus status;

    private final long computeTokens;

    private final WorkerEndpoint.RequestRegistration registration;
    private final SelectionRollback registrationOwner;

    private RoleSelection(
            WorkerEndpoint endpoint,
            ServerStatus status,
            long computeTokens,
            WorkerEndpoint.RequestRegistration registration,
            SelectionRollback registrationOwner) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        checkArgument(status.isSuccess(), "route requires successful Worker metadata");
        this.status = ServerStatus.copyOf(status);
        this.computeTokens = computeTokens;
        this.registration = registration;
        this.registrationOwner = registrationOwner;
    }

    public final WorkerEndpoint endpoint() {
        return endpoint;
    }

    public final long dpRank() {
        return status.getDpRank();
    }

    public final long hitCacheTokens() {
        return status.getDebugInfo() == null ? 0L : status.getDebugInfo().getHitCacheLen();
    }

    /**
     * Builds a fresh protocol projection; callers cannot mutate the selected facts.
     */
    public final ServerStatus toServerStatus(String requestId) {
        var response = ServerStatus.copyOf(status);
        response.setRequestId(requestId);
        return response;
    }

    public static RoleSelection prefill(
            PrefillEndpoint endpoint,
            ServerStatus status,
            long computeTokens,
            PrefillEndpoint.RequestRegistration registration) {
        checkArgument(
                status.getRole() != null && status.getRole().supportsPrefill(),
                "Prefill role required");
        checkArgument(computeTokens >= 0L, "Prefill work must be non-negative");
        return new RoleSelection(endpoint, status, computeTokens, registration, endpoint);
    }

    public static RoleSelection decode(
            DecodeEndpoint endpoint,
            ServerStatus status,
            DecodeEndpoint.RequestRegistration registration) {
        checkArgument(status.getRole() == RoleType.DECODE, "Decode role required");
        checkArgument(
                registration == null
                        || Objects.equals(registration.requestId(), status.getRequestId()),
                "Decode reservation belongs to another request");
        return new RoleSelection(endpoint, status, 0L, registration, endpoint);
    }

    public static RoleSelection encoder(
            EncoderEndpoint endpoint,
            ServerStatus status,
            EncoderEndpoint.RequestRegistration registration) {
        checkArgument(status.getRole() == RoleType.ENCODER, "Encoder role required");
        checkArgument(
                registration == null || registration.requestId().equals(status.getRequestId()),
                "Encoder registration belongs to another request");
        return new RoleSelection(endpoint, status, 0L, registration, endpoint);
    }

    public static RoleSelection auxiliary(WorkerEndpoint endpoint, ServerStatus status) {
        checkArgument(status.getRole() == RoleType.VIT, "VIT role required");
        return new RoleSelection(endpoint, status, 0L, null, null);
    }

    public WorkerEndpoint.RequestRegistration registration() {
        return registration;
    }

    public long computeTokens() {
        return computeTokens;
    }

    public final CapacityKey capacityKey() {
        return CapacityKey.exact(role(), group(), status.getLogicalIpPort());
    }

    public final RoleType role() {
        return status.getRole();
    }

    public final String group() {
        return status.getGroup();
    }

    public String requestId() {
        return status.getRequestId();
    }

    public static boolean releaseDecode(
            DecodeEndpoint endpoint,
            DecodeEndpoint.RequestRegistration reservation,
            WorkerEndpoint.ReleaseReason reason) {
        if (endpoint == null || reservation == null) return true;
        var state = endpoint.registrationState(reservation);
        if (reason == WorkerEndpoint.ReleaseReason.COUNTERPART_FINISHED
                && state != DecodeResources.RegistrationState.UNCONFIRMED
                && state != DecodeResources.RegistrationState.NOT_REGISTERED) return false;
        if (reason == WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK) {
            endpoint.rollbackSelection(reservation);
            return true;
        }
        endpoint.releaseRequest(reservation);
        return true;
    }

    final String ip() {
        return status.getServerIp();
    }

    final int httpPort() {
        return status.getHttpPort();
    }

    final int grpcPort() {
        return status.getGrpcPort();
    }

    void rollbackSelection() {
        if (registration != null) registrationOwner.rollbackSelection(registration);
    }
}
