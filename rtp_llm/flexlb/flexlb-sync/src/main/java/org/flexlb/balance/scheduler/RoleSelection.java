package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * One selected role and its exact registration. Metadata is frozen; the Endpoint owns live capacity.
 * The same registration is retained from selection through queueing and delivery.
 */
public abstract sealed class RoleSelection {
    private final WorkerEndpoint endpoint;
    private final ServerStatus status;

    private RoleSelection(WorkerEndpoint endpoint, ServerStatus status) {
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
        checkArgument(status.isSuccess(), "route requires successful Worker metadata");
        checkArgument(
                status.getRole() != null && status.getRole() != RoleType.FRONTEND,
                "unsupported Worker role");
        this.status = ServerStatus.copyOf(status);
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

    public final PlacementKey placementKey() {
        return PlacementKey.exact(role(), group(), status.getLogicalIpPort());
    }

    public final RoleType role() {
        return status.getRole();
    }

    public final String group() {
        return status.getGroup();
    }

    /**
     * Builds a fresh protocol projection; callers cannot mutate the selected facts.
     */
    public final ServerStatus toServerStatus(String requestId) {
        var response = ServerStatus.copyOf(status);
        response.setRequestId(requestId);
        return response;
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

    String requestId() {
        return status.getRequestId();
    }

    /**
     * Retains the selected Prefill endpoint, predicted work and exact resource registration.
     */
    public static final class Prefill extends RoleSelection {
        private final long workMs;
        final PrefillEndpoint.RequestRecord registration;

        Prefill(
                PrefillEndpoint endpoint,
                ServerStatus status,
                long workMs,
                PrefillEndpoint.RequestRecord registration) {
            super(endpoint, status);
            checkArgument(status.getRole().supportsPrefill(), "Prefill role required");
            checkArgument(workMs >= 0L, "Prefill work must be non-negative");
            this.workMs = workMs;
            this.registration = registration;
        }

        public long workMs() {
            return workMs;
        }
    }

    /**
     * Retains the selected Decode endpoint and validates and releases its exact registration.
     */
    public static final class Decode extends RoleSelection {
        final org.flexlb.balance.endpoint.DecodeResources.RequestReservation registration;

        Decode(
                DecodeEndpoint endpoint,
                ServerStatus status,
                org.flexlb.balance.endpoint.DecodeResources.RequestReservation registration) {
            super(endpoint, status);
            checkArgument(status.getRole() == RoleType.DECODE, "Decode role required");
            checkArgument(
                    registration == null || Objects.equals(registration.requestId(), requestId()),
                    "Decode reservation belongs to another request");
            this.registration = registration;
        }

        static boolean release(
                DecodeEndpoint endpoint,
                DecodeResources.RequestReservation reservation,
                DecodeResources.ReleaseReason reason) {
            if (endpoint == null || reservation == null) return true;
            var released = endpoint.release(reservation, reason).status();
            boolean complete =
                    released == DecodeResources.ReservationReleaseResult.RELEASED
                            || released == DecodeResources.ReservationReleaseResult.STALE;
            checkState(
                    complete
                            || reason != DecodeResources.ReleaseReason.LOCAL_ROLLBACK
                                    && reason
                                            != DecodeResources.ReleaseReason.UNSENT_ROUTE_ROLLBACK,
                    "route rollback retained Decode resources: %s",
                    released);
            return complete;
        }
    }

    /**
     * Encoder and VIT select an endpoint without registering P/D capacity.
     */
    public static final class Auxiliary extends RoleSelection {
        public Auxiliary(WorkerEndpoint endpoint, ServerStatus status) {
            super(endpoint, status);
            checkArgument(
                    status.getRole() == RoleType.VIT
                            || status.getRole() == RoleType.ENCODER
                                    && endpoint instanceof EncoderEndpoint,
                    "Encoder or VIT role required");
        }
    }
}
