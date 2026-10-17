package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.util.Failures;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Builds one route in a single generation-pin scope. Selection writes frozen facts
 * here; Selector closes failures and publishes this same route after prepare().
 * Capacity creates a new callback identity sharing these pins. Closing releases only
 * pins; Scheduler retains responsibility for committed resource cleanup.
 */
public final class RequestRoute implements GroupPlanner.Input, AutoCloseable {

    private static final byte BUILDING = 0, PREPARED = 1, ABORTED = 2;
    // Written only during thread-confined construction; closing a published route changes only its pins.
    private byte constructionState = BUILDING;
    private final RequestContext ctx;
    private Response routeResponse;
    private WorkerEndpoint.GenerationPin[] pins;
    // Role slots in the three-pin scope; -1 means that role has not been selected.
    private byte prefillIndex = -1;
    private byte decodeIndex = -1;
    private ServerStatus prefill;
    private ServerStatus decode;
    private ServerStatus vit;
    private long prefillWorkMs;
    private long prefillVersion;
    private long decodeVersion;
    private final DecodeResources.ReservationHandle decodeReservation;

    /** Construction is thread confined; prepare publishes the completed route. */
    public RequestRoute(RequestContext context) {
        this(context, null, null);
    }

    private RequestRoute(RequestContext context, RequestRoute selected,
                         DecodeResources.ReservationHandle reservation) {
        ctx = Objects.requireNonNull(context, "context");
        decodeReservation = reservation;
        if (selected != null) {
            constructionState = PREPARED;
            routeResponse = selected.routeResponse;
            pins = selected.pins;
            prefillIndex = selected.prefillIndex;
            decodeIndex = selected.decodeIndex;
            prefill = selected.prefill;
            decode = selected.decode;
            prefillWorkMs = selected.prefillWorkMs;
            prefillVersion = selected.prefillVersion;
            decodeVersion = selected.decodeVersion;
        }
    }

    /** Consumes the pin even on validation failure; only selection may add facts. */
    public ServerStatus add(WorkerEndpoint.GenerationPin pin, ServerStatus status, long workMs, long version) {
        try {
            checkState(constructionState == BUILDING, "route selection is not building");
            checkState(pin != null && pin.isOpen(), "selected endpoint generation is closed");
            ServerStatus frozen = ServerStatus.copyOf(Objects.requireNonNull(status, "worker metadata"));
            checkArgument(frozen.isSuccess(), "route requires successful Worker metadata");
            checkState(frozen.getRequestId() == requestId(), "selected Worker belongs to another request");
            WorkerEndpoint endpoint = pin.endpoint();
            checkArgument(Objects.equals(frozen.getServerIp(), endpoint.getIp())
                    && frozen.getHttpPort() == endpoint.getHttpPort(), "selection metadata does not match pinned endpoint address");
            checkArgument(version >= 0L, "placementVersion must be non-negative");
            RoleType role = frozen.getRole();
            if (role != null && role.supportsPrefill()) {
                checkState(prefillIndex < 0 && endpoint instanceof PrefillEndpoint,
                        "request requires one exact Prefill assignment");
                checkArgument(workMs >= 0L, "Prefill work must be non-negative");
            } else if (role == RoleType.DECODE) {
                checkState(decodeIndex < 0 && endpoint instanceof DecodeEndpoint,
                        "request requires at most one exact Decode assignment");
            } else {
                checkArgument(role == RoleType.VIT, "unsupported Worker role");
                checkState(vit == null, "request requires at most one VIT assignment");
            }
            if (pins == null) { pins = new WorkerEndpoint.GenerationPin[3]; }
            int index = 0;
            while (index < pins.length && pins[index] != null) { index++; }
            checkState(index < pins.length, "route supports at most three Worker roles");
            pins[index] = pin;
            if (role.supportsPrefill()) {
                prefillIndex = (byte) index;
                prefill = frozen;
                prefillWorkMs = workMs;
                prefillVersion = version;
            } else if (role == RoleType.DECODE) {
                decodeIndex = (byte) index;
                decode = frozen;
                decodeVersion = version;
            } else { vit = frozen; }
            return status;
        } catch (RuntimeException | Error failure) {
            if (constructionState == BUILDING) { constructionState = ABORTED; }
            if (pin != null) { Failures.append(failure, Failures.close(pin)); }
            throw failure;
        }
    }

    /** Finish construction once; the selector retains cleanup responsibility on failure. */
    RequestRoute prepare() {
        checkState(constructionState == BUILDING, "route selection is not building");
        constructionState = ABORTED; // Every unsuccessful completion leaves this construction unusable.
        checkState(prefillIndex >= 0, "selection must complete its own Prefill route");
        for (var pin : pins) {
            checkState(pin == null || pin.isOpen(), "selected endpoint generation is closed");
        }
        var statuses = new ArrayList<ServerStatus>(3);
        for (int i = 0; i < pins.length; i++) {
            var pin = pins[i];
            if (pin == null) { continue; }
            statuses.add(i == prefillIndex ? prefill : i == decodeIndex ? decode : vit);
            if (i != prefillIndex && i != decodeIndex) {
                pins[i] = null;
                pin.close();
            }
        }
        Response response = new Response();
        response.setServerStatus(statuses);
        response.setSuccess(true);
        routeResponse = response;
        vit = null;
        constructionState = PREPARED;
        return this;
    }

    /** Attach exact capacity without copying assignments or mutating an earlier route identity. */
    static RequestRoute create(RequestContext context, RequestRoute selected,
                               DecodeResources.ReservationHandle reservation) {
        checkState(selected.constructionState == PREPARED, "capacity requires a prepared route");
        var inputs = Objects.requireNonNull(context.data().getRequest(), "request inputs");
        Objects.requireNonNull(context.data().getSchedulingMetadata(), "scheduling metadata");
        checkArgument(context.data().getRequestId() == selected.requestId(), "assignment belongs to another request");
        checkArgument(reservation == null || reservation.requestId() == inputs.getRequestId(),
                "Decode reservation belongs to another request");
        return new RequestRoute(context, selected, reservation);
    }

    WorkerEndpoint blockedEndpointIfCurrent(PlacementKey blocker) {
        boolean isDecode = blocker.role() == RoleType.DECODE;
        checkArgument((isDecode ? decodePin() : prefillPin()) != null && blocker.equals(placementKey(blocker.role())),
                "blocker does not belong to this Worker assignment");
        WorkerEndpoint endpoint = isDecode ? decodeEp() : prefillEp();
        long version = endpoint instanceof PrefillEndpoint worker ? worker.ownership.progress().ownershipVersion()
                : ((DecodeEndpoint) endpoint).placementVersion();
        return version == (isDecode ? decodeVersion : prefillVersion) ? endpoint : null;
    }

    PlacementKey placementKey(RoleType role) {
        ServerStatus status = role == RoleType.DECODE ? decode : prefill;
        WorkerEndpoint endpoint = role == RoleType.DECODE ? decodeEp() : prefillEp();
        return PlacementKey.exact(status.getRole(), status.getGroup(), endpoint.ipPort());
    }

    WorkerEndpoint.GenerationPin prefillPin() { return prefillIndex < 0 ? null : pins[prefillIndex]; }
    WorkerEndpoint.GenerationPin decodePin() { return decodeIndex < 0 ? null : pins[decodeIndex]; }
    public long prefillWorkMs() { return prefillWorkMs; }

    public RequestContext ctx() { return ctx; }

    /** O(1) bound; grouping must not parse or copy serialized GenerateInput payloads. */
    public long batchPayloadSizeUpperBound() {
        return ctx.data().getGenerateInputSize() + 64L
                + roleAddressSizeUpperBound(prefill) + roleAddressSizeUpperBound(decode);
    }

    private static long roleAddressSizeUpperBound(ServerStatus status) {
        if (status == null) { return 0L; }
        String ip = status.getServerIp();
        return 64L + (ip == null ? 0L : 3L * ip.length());
    }
    public CompletableFuture<Response> future() { return ctx.data().getFuture(); }
    public Response successResponse(boolean enqueuedByMaster) {
        return Response.buildSuccessResponse(routeResponse, enqueuedByMaster);
    }
    public ServerStatus prefill() { return ServerStatus.copyOf(prefill); }
    public ServerStatus decode() { return ServerStatus.copyOf(decode); }
    public long prefillDpRank() { return prefill.getDpRank(); }
    public PrefillEndpoint prefillEp() { return (PrefillEndpoint) prefillPin().endpoint(); }
    public DecodeEndpoint decodeEp() { return decodeIndex < 0 ? null : (DecodeEndpoint) decodePin().endpoint(); }
    public DecodeResources.ReservationHandle decodeReservation() { return decodeReservation; }

    public long enqueuedAtMs() { return ctx.data().getFirstWorkerEnqueueTime(); }
    public long expiresAtMs() { return ctx.data().getRequestExpiresAtMs(); }
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
    public long hitCache() { return prefill.getDebugInfo() == null ? 0L : prefill.getDebugInfo().getHitCacheLen(); }

    @Override
    public void close() {
        if (constructionState == BUILDING) { constructionState = ABORTED; }
        if (pins == null) { return; }
        Throwable failure = null;
        for (int i = pins.length - 1; i >= 0; i--) {
            var pin = pins[i];
            if (pin != null && pin.isOpen()) { failure = Failures.append(failure, Failures.close(pin)); }
        }
        Failures.rethrow(failure, "route pin cleanup failed");
    }
}
