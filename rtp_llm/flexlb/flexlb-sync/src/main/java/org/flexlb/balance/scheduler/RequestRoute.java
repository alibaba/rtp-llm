package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Builds one route in a single generation-pin scope. Selection writes frozen facts
 * here; Selector closes failures and publishes this same route after prepare().
 * Capacity attaches to this same identity before publication. Closing releases only
 * pins; RequestLifecycle remains responsible for committed resource cleanup.
 */
public final class RequestRoute
        implements GroupPlanner.Input, PrefillRequestQueue.QueueItem, AutoCloseable {

    private static final byte BUILDING = 0, PREPARED = 1, ABORTED = 2, RESERVED = 3;
    // Written only during thread-confined construction; closing a published route changes only its
    // pins.
    private byte constructionState = BUILDING;
    private final RequestContext ctx;
    private Response routeResponse;
    private WorkerEndpoint.GenerationPin prefillPin, decodePin, vitPin;
    private ArrayList<ServerStatus> assignments;
    private ServerStatus prefill;
    private ServerStatus decode;
    private long prefillWorkMs;
    private long prefillVersion;
    private long decodeVersion;
    private DecodeResources.ReservationHandle decodeReservation;
    private PrefillEndpoint.RouteReservation prefillReservation;

    // Cached immutable planning input; queue implementations only store request identities.
    private volatile GroupPlanner.Item planningInput;

    public GroupPlanner.Item planningInput() {
        var result = planningInput;
        if (result != null) return result;
        synchronized (this) {
            if (planningInput == null) {
                planningInput =
                        new GroupPlanner.Item(
                                requestId(),
                                priority(),
                                enqueueSeq(),
                                enqueuedAtMs(),
                                expiresAtMs(),
                                seqLen(),
                                hitCache());
            }
            return planningInput;
        }
    }

    /** Construction and capacity attachment finish before this identity is published to Context. */
    public RequestRoute(RequestContext context) {
        ctx = Objects.requireNonNull(context, "context");
    }

    /** Consumes the pin even on validation failure; only selection may add facts. */
    public ServerStatus add(
            WorkerEndpoint.GenerationPin pin, ServerStatus status, long workMs, long version) {
        try {
            checkState(constructionState == BUILDING, "route selection is not building");
            ServerStatus frozen = validateSelection(pin, status, version);
            WorkerEndpoint endpoint = pin.endpoint();
            RoleType role = frozen.getRole();
            if (role != null && role.supportsPrefill()) {
                checkState(
                        prefillPin == null && endpoint instanceof PrefillEndpoint,
                        "request requires one exact Prefill assignment");
                checkArgument(workMs >= 0L, "Prefill work must be non-negative");
                prefillPin = pin;
                prefill = frozen;
                prefillWorkMs = workMs;
                prefillVersion = version;
            } else if (role == RoleType.DECODE) {
                checkState(
                        decodePin == null && endpoint instanceof DecodeEndpoint,
                        "request requires at most one exact Decode assignment");
                decodePin = pin;
                decode = frozen;
                decodeVersion = version;
            } else {
                checkArgument(role == RoleType.VIT, "unsupported Worker role");
                checkState(vitPin == null, "request requires at most one VIT assignment");
                vitPin = pin;
            }
            if (assignments == null) assignments = new ArrayList<>(3);
            assignments.add(frozen);
            return status;
        } catch (RuntimeException | Error failure) {
            if (constructionState == BUILDING) {
                constructionState = ABORTED;
            }
            if (pin != null) {
                Failures.append(failure, Failures.close(pin));
            }
            throw failure;
        }
    }

    private ServerStatus validateSelection(
            WorkerEndpoint.GenerationPin pin, ServerStatus status, long version) {
        checkState(pin != null && pin.isOpen(), "selected endpoint generation is closed");
        ServerStatus frozen =
                ServerStatus.copyOf(Objects.requireNonNull(status, "worker metadata"));
        checkArgument(frozen.isSuccess(), "route requires successful Worker metadata");
        checkState(
                frozen.getRequestId() == requestId(), "selected Worker belongs to another request");
        WorkerEndpoint endpoint = pin.endpoint();
        checkArgument(
                Objects.equals(frozen.getServerIp(), endpoint.getIp())
                        && frozen.getHttpPort() == endpoint.getHttpPort(),
                "selection metadata does not match pinned endpoint address");
        checkArgument(version >= 0L, "placementVersion must be non-negative");
        return frozen;
    }

    /** Finish construction once; the selector retains cleanup responsibility on failure. */
    RequestRoute prepare() {
        checkState(constructionState == BUILDING, "route selection is not building");
        constructionState =
                ABORTED; // Every unsuccessful completion leaves this construction unusable.
        checkState(prefillPin != null, "selection must complete its own Prefill route");
        checkState(
                prefillPin.isOpen()
                        && (decodePin == null || decodePin.isOpen())
                        && (vitPin == null || vitPin.isOpen()),
                "selected endpoint generation is closed");
        if (vitPin != null) {
            var selectedVit = vitPin;
            vitPin = null;
            selectedVit.close();
        }
        Response response = new Response();
        response.setServerStatus(assignments);
        response.setSuccess(true);
        routeResponse = response;
        constructionState = PREPARED;
        return this;
    }

    /** Attach the acquired capacity once, before queue or delivery publication. */
    RequestRoute attachReservation(DecodeResources.ReservationHandle reservation) {
        if (constructionState == RESERVED) {
            checkState(
                    Objects.equals(decodeReservation, reservation),
                    "route capacity identity already assigned");
            return this;
        }
        checkState(
                constructionState == PREPARED, "capacity requires an unpublished prepared route");
        var inputs = Objects.requireNonNull(ctx.data().getRequest(), "request inputs");
        Objects.requireNonNull(ctx.data().getSchedulingMetadata(), "scheduling metadata");
        checkArgument(
                reservation == null || reservation.requestId() == inputs.getRequestId(),
                "Decode reservation belongs to another request");
        decodeReservation = reservation;
        constructionState = RESERVED;
        return this;
    }

    WorkerEndpoint blockedEndpointIfCurrent(PlacementKey blocker) {
        boolean isDecode = blocker.role() == RoleType.DECODE;
        checkArgument(
                (isDecode ? decodePin() : prefillPin()) != null
                        && blocker.equals(placementKey(blocker.role())),
                "blocker does not belong to this Worker assignment");
        WorkerEndpoint endpoint = isDecode ? decodeEp() : prefillEp();
        long version =
                endpoint instanceof PrefillEndpoint worker
                        ? worker.progress().ownershipVersion()
                        : ((DecodeEndpoint) endpoint).placementVersion();
        return version == (isDecode ? decodeVersion : prefillVersion) ? endpoint : null;
    }

    PlacementKey placementKey(RoleType role) {
        ServerStatus status = role == RoleType.DECODE ? decode : prefill;
        WorkerEndpoint endpoint = role == RoleType.DECODE ? decodeEp() : prefillEp();
        return PlacementKey.exact(status.getRole(), status.getGroup(), endpoint.ipPort());
    }

    WorkerEndpoint.GenerationPin prefillPin() {
        return prefillPin;
    }

    WorkerEndpoint.GenerationPin decodePin() {
        return decodePin;
    }

    public long prefillWorkMs() {
        return prefillWorkMs;
    }

    public RequestContext ctx() {
        return ctx;
    }

    public Response successResponse(boolean enqueuedByMaster) {
        return Response.buildSuccessResponse(routeResponse, enqueuedByMaster);
    }

    public ServerStatus prefill() {
        return ServerStatus.copyOf(prefill);
    }

    public ServerStatus decode() {
        return ServerStatus.copyOf(decode);
    }

    public long prefillDpRank() {
        return prefill.getDpRank();
    }

    public PrefillEndpoint prefillEp() {
        return (PrefillEndpoint) prefillPin().endpoint();
    }

    public DecodeEndpoint decodeEp() {
        return decodePin == null ? null : (DecodeEndpoint) decodePin.endpoint();
    }

    /** Exact Prefill resource record; its presence records the physical handoff before callbacks. */
    public PrefillEndpoint.RouteReservation prefillReservation() {
        return prefillReservation;
    }

    public void attachPrefillReservation(PrefillEndpoint.RouteReservation reservation) {
        checkArgument(reservation.belongsTo(this), "Prefill reservation belongs to another route");
        prefillReservation = reservation;
    }

    public DecodeResources.ReservationHandle decodeReservation() {
        return decodeReservation;
    }

    public long enqueuedAtMs() {
        return ctx.data().getFirstWorkerEnqueueTime();
    }

    public long expiresAtMs() {
        return ctx.data().getRequestExpiresAtMs();
    }

    public boolean requestExpired(long nowMs) {
        long expiry = expiresAtMs();
        return expiry <= 0L || nowMs >= expiry;
    }

    /** Normalized priority for the Worker queue index. */
    public int priority() {
        return ctx.data().getPriority();
    }

    /** Stable FIFO sequence and same-priority tie-break; retained on re-offer. */
    public long enqueueSeq() {
        return ctx.data().getWorkerEnqueueSequence();
    }

    // -- derived accessors --

    public long requestId() {
        return ctx.data().getRequestId();
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return ctx.data().getRequest().getHardKvTokens();
    }

    /** Frozen cache-hit tokens on the assigned Prefill generation. */
    public long hitCache() {
        return prefill.getDebugInfo() == null ? 0L : prefill.getDebugInfo().getHitCacheLen();
    }

    @Override
    public void close() {
        if (constructionState == BUILDING) {
            constructionState = ABORTED;
        }
        if (assignments == null) return;
        Throwable failure = null;
        for (int i = assignments.size() - 1; i >= 0; i--) {
            var role = assignments.get(i).getRole();
            var pin =
                    role.supportsPrefill()
                            ? prefillPin
                            : role == RoleType.DECODE ? decodePin : vitPin;
            if (pin != null && pin.isOpen()) {
                failure = Failures.append(failure, Failures.close(pin));
            }
        }
        Failures.rethrow(failure, "route pin cleanup failed");
    }
}
