package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

/**
 * One registered role combination and the exact route identity used by queues and callbacks.
 * Selection binds its Prefill registration before publication. RequestContext owns publication
 * and delivery state; this object retains only the request association and frozen role results.
 */
public final class RouteResult implements GroupPlanner.Input, PrefillRequestQueue.QueueItem {
    private static final java.util.List<RoleType> ENDPOINT_CHECK_ORDER =
            java.util.List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.ENCODER, RoleType.VIT);

    private final RequestContext ctx;
    private final String requestId;
    private final List<RoleSelection> roles;

    public RouteResult(RequestContext context, List<RoleSelection> roles) {
        this.ctx = Objects.requireNonNull(context, "context");
        this.requestId = Objects.requireNonNull(context.data().getRequestId(), "requestId");
        this.roles = List.copyOf(roles);
        var seen = java.util.EnumSet.noneOf(RoleType.class);
        for (var selection : this.roles) {
            checkState(
                    Objects.equals(requestId, selection.requestId()),
                    "selected Worker belongs to another request");
            checkState(
                    seen.add(
                            selection.role().supportsPrefill()
                                    ? RoleType.PREFILL
                                    : selection.role()),
                    "request requires at most one assignment per role");
        }
        checkState(
                seen.contains(RoleType.PREFILL) || seen.contains(RoleType.ENCODER),
                "one Prefill or Encoder route required");
        var prefill = prefillReservation();
        if (prefill != null) prefill.bind(this);
    }

    @Override
    public String requestId() {
        return requestId;
    }

    public List<RoleSelection> roles() {
        return roles;
    }

    public Response successResponse(boolean enqueuedByMaster) {
        var response = new Response();
        var statuses = new ArrayList<org.flexlb.dao.loadbalance.ServerStatus>(roles.size());
        for (var role : roles) statuses.add(role.toServerStatus(requestId));
        response.setServerStatus(statuses);
        response.setSuccess(true);
        response.setCode(200);
        response.setEnqueuedByMaster(enqueuedByMaster);
        return response;
    }

    public GroupPlanner.Item planningInput() {
        return prefillReservation().planningInput();
    }

    /**
     * Exact Prefill resource record acquired during selection; presence does not imply publication.
     */
    public PrefillEndpoint.RequestRecord prefillReservation() {
        var prefill = (RoleSelection.Prefill) role(RoleType.PREFILL);
        return prefill == null ? null : prefill.registration;
    }

    public RoleSelection role(RoleType role) {
        for (var selection : roles) {
            if (selection.role() == role
                    || role.supportsPrefill() && selection.role().supportsPrefill())
                return selection;
        }
        return null;
    }

    public EncoderEndpoint encoderEndpoint() {
        return (EncoderEndpoint) endpoint(RoleType.ENCODER);
    }

    public long prefillWorkMs() {
        var prefill = (RoleSelection.Prefill) role(RoleType.PREFILL);
        return prefill == null ? 0L : prefill.workMs();
    }

    public RequestContext ctx() {
        return ctx;
    }

    public long prefillDpRank() {
        return role(RoleType.PREFILL).dpRank();
    }

    public PrefillEndpoint prefillEndpoint() {
        return (PrefillEndpoint) endpoint(RoleType.PREFILL);
    }

    public DecodeEndpoint decodeEndpoint() {
        return (DecodeEndpoint) endpoint(RoleType.DECODE);
    }

    public DecodeResources.RequestReservation decodeReservation() {
        var decode = (RoleSelection.Decode) role(RoleType.DECODE);
        return decode == null ? null : decode.registration;
    }

    public long enqueuedAtMs() {
        return ctx.data().getFirstWorkerEnqueueTime();
    }

    public boolean requestExpired(long nowMs) {
        long expiry = expiresAtMs();
        return expiry <= 0L || nowMs >= expiry;
    }

    public long expiresAtMs() {
        return ctx.data().getRequestExpiresAtMs();
    }

    /**
     * Normalized priority for the Worker queue index.
     */
    public int priority() {
        return ctx.data().getPriority();
    }

    /**
     * Stable FIFO sequence and same-priority tie-break; retained on re-offer.
     */
    public long enqueueSeq() {
        return ctx.data().getWorkerEnqueueSequence();
    }

    /**
     * Total sequence length of this request.
     */
    public long seqLen() {
        return ctx.data().getRequest().getHardKvTokens();
    }

    /**
     * Frozen cache-hit tokens on the assigned Prefill generation.
     */
    public long hitCache() {
        return role(RoleType.PREFILL).hitCacheTokens();
    }

    /**
     * Validate a selection before publication, or classify a rejected admission after it.
     * An unavailable endpoint invalidates the whole selection, regardless of its role.
     * Delivered routes must retain their delivery result instead of being validated again.
     */
    <T> RoutingResult<T, CapacityKey> validateEndpoints(RoutingResult<T, CapacityKey> result) {
        return validateEndpoints(roles, result);
    }

    static <T> RoutingResult<T, CapacityKey> validateEndpoints(
            List<RoleSelection> roles, RoutingResult<T, CapacityKey> result) {
        for (var role : roles) {
            if (!role.endpoint().acceptsNewRequests())
                return RoutingResult.retry(
                        Response.buildErrorResponse(
                                role.role().getErrorType(),
                                "Selected "
                                        + role.role()
                                        + " endpoint became unavailable: "
                                        + role.endpoint().getIp()
                                        + (result.failure() == null
                                                ? ""
                                                : "; preceding failure: "
                                                        + result.failure().getErrorMessage())));
        }
        return result;
    }

    boolean selectedEndpointsAvailable() {
        return firstUnavailableEndpoint() == null;
    }

    WorkerEndpoint firstUnavailableEndpoint() {
        for (RoleType role : ENDPOINT_CHECK_ORDER) {
            var endpoint = endpoint(role);
            if (endpoint != null && !endpoint.acceptsNewRequests()) return endpoint;
        }
        return null;
    }

    WorkerEndpoint endpoint(RoleType role) {
        var selected = role(role);
        return selected == null ? null : selected.endpoint();
    }

    boolean usesAnyEndpoint(java.util.Set<WorkerEndpoint> endpoints) {
        for (var selected : roles()) {
            if (endpoints.contains(selected.endpoint())) return true;
        }
        return false;
    }

    CapacityKey capacityKey(RoleType role) {
        return role(role).capacityKey();
    }
}
