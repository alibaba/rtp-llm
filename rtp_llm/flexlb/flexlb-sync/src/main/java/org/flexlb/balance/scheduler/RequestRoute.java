package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * One request's exact Worker assignments and optional Decode reservation identity.
 * Assignment facts are shared unchanged when capacity is acquired. The resulting
 * route identity fences callbacks; Scheduler owns rollback until its handoff succeeds.
 * Closing ends only the generation-pin scope, never the committed resource lifetime.
 */
public final class RequestRoute implements GroupPlanner.Input, AutoCloseable {

    private final RequestContext ctx;
    private final Response routeResponse;
    private final WorkerAssignment prefill;
    private final WorkerAssignment decode;
    private final DecodeResources.ReservationHandle decodeReservation;

    private RequestRoute(RequestContext ctx, Response routeResponse, WorkerAssignment prefill,
                         WorkerAssignment decode, DecodeResources.ReservationHandle decodeReservation) {
        this.ctx = Objects.requireNonNull(ctx, "context");
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.decodeReservation = decodeReservation;
    }

    /** The caller owns every pin until the complete assignment is returned. */
    static RequestRoute prepare(RequestContext context, List<WorkerAssignment> assignments) {
        List<ServerStatus> statuses = new ArrayList<>(assignments.size());
        for (WorkerAssignment assigned : assignments) { statuses.add(assigned.serverStatus()); }
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(statuses);
        WorkerAssignment prefill = null;
        WorkerAssignment decode = null;
        for (WorkerAssignment assigned : assignments) {
            checkState(assigned.requestId() == context.getRequestId(), "assigned Worker belongs to another request");
            RoleType role = assigned.role();
            if (role != null && role.supportsPrefill()) {
                checkState(prefill == null && assigned.endpoint() instanceof PrefillEndpoint,
                        "request requires one exact Prefill assignment");
                assigned.prefillWorkMs();
                prefill = assigned;
            } else if (role == RoleType.DECODE) {
                checkState(decode == null && assigned.endpoint() instanceof DecodeEndpoint,
                        "request requires at most one exact Decode assignment");
                decode = assigned;
            } else {
                assigned.close();
            }
        }
        checkState(prefill != null, "request has no Prefill endpoint generation");
        RequestRoute route = new RequestRoute(context,
                response, prefill, decode, null);
        prefill.assignToRequest();
        if (decode != null) { decode.assignToRequest(); }
        return route;
    }

    /** Attach exact capacity without copying assignments or mutating an earlier route identity. */
    static RequestRoute create(RequestContext context, RequestRoute selected,
                               DecodeResources.ReservationHandle reservation) {
        RequestRequirements inputs = Objects.requireNonNull(context.getRequirements(), "registered request inputs");
        checkArgument(context.getRequestId() == selected.requestId(), "assignment belongs to another request");
        checkArgument(reservation == null || reservation.requestId() == inputs.requestId(),
                "Decode reservation belongs to another request");
        return new RequestRoute(context, selected.routeResponse, selected.prefill, selected.decode, reservation);
    }

    WorkerEndpoint blockedEndpointIfCurrent(PlacementKey blocker) {
        WorkerAssignment assigned = blocker.role() == RoleType.DECODE ? decode : prefill;
        checkArgument(assigned != null && blocker.equals(placementKey(assigned)),
                "blocker does not belong to this Worker assignment");
        WorkerEndpoint endpoint = assigned.endpoint();
        long version = endpoint instanceof PrefillEndpoint worker ? worker.placementVersion()
                : ((DecodeEndpoint) endpoint).placementVersion();
        return version == assigned.placementVersion() ? endpoint : null;
    }

    private static PlacementKey placementKey(WorkerAssignment assigned) {
        return PlacementKey.exact(assigned.role(), assigned.group(), assigned.endpoint().ipPort());
    }

    PlacementKey prefillPlacementKey() { return placementKey(prefill); }
    PlacementKey decodePlacementKey() { return placementKey(decode); }
    WorkerEndpoint.GenerationPin prefillPin() { return prefill.generationPin(); }
    WorkerEndpoint.GenerationPin decodePin() { return decode == null ? null : decode.generationPin(); }
    long prefillWorkMs() { return prefill.prefillWorkMs(); }

    public RequestContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return ctx.getFuture(); }
    public Response successResponse(boolean enqueuedByMaster) {
        return Response.buildSuccessResponse(routeResponse, enqueuedByMaster);
    }
    public ServerStatus prefill() { return prefill.serverStatus(); }
    public ServerStatus decode() { return decode == null ? null : decode.serverStatus(); }
    public PrefillEndpoint prefillEp() { return (PrefillEndpoint) prefill.endpoint(); }
    public DecodeEndpoint decodeEp() { return decode == null ? null : (DecodeEndpoint) decode.endpoint(); }
    public DecodeResources.ReservationHandle decodeReservation() { return decodeReservation; }

    public RequestRequirements requirements() {
        return ctx.getRequirements();
    }
    public long enqueuedAtMs() { return ctx.getFirstWorkerEnqueueTime(); }
    public long expiresAtMs() { return ctx.getRequestExpiresAtMs(); }
    public boolean requestExpired(long nowMs) {
        return ctx.requestExpired(nowMs);
    }

    /** Whether ACTIVE publication must atomically own one route reservation. */
    public boolean requiresRouteReservation() {
        return requirements().requiresRouteReservation();
    }

    /** Normalized priority for the Worker queue index. */
    public int priority() {
        return requirements().priority();
    }

    /** Stable FIFO sequence and same-priority tie-break; retained on re-offer. */
    public long enqueueSeq() {
        return ctx.getWorkerEnqueueSequence();
    }

    // -- derived accessors --

    public long requestId() {
        return requirements().requestId();
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return requirements().hardKvTokens();
    }

    /** Frozen cache-hit tokens on the assigned Prefill generation. */
    public long hitCache() { return prefill.hitCache(); }

    @Override
    public void close() {
        try (prefill; decode) {
            // Acquired resources retain their existing Scheduler/Context cleanup responsibility.
        }
    }
}
