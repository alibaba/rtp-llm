package org.flexlb.balance.schedule;

import org.flexlb.balance.request.Response;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.endpoint.WorkerRoute;
import org.flexlb.worker.status.ServerStatus;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Holds the selected roles and exact resource registrations for one request route.
 * Delivery, worker feedback and cleanup share this object's identity.
 */
@lombok.Getter
@lombok.experimental.Accessors(fluent = true)
public final class RequestRoute implements WorkerRoute {
    private static final List<RoleType> ENDPOINT_CHECK_ORDER =
            List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.ENCODER, RoleType.VIT);
    private final RequestContext ctx;
    private final String requestId;
    private final List<RoleSelection> roles;

    public RequestRoute(RequestContext context, List<RoleSelection> roles) {
        this.ctx = Objects.requireNonNull(context, "context");
        this.requestId = Objects.requireNonNull(context.getRequestId(), "requestId");
        this.roles = List.copyOf(roles);
        var seen = java.util.EnumSet.noneOf(RoleType.class);
        for (var selection : this.roles) {
            checkArgument(
                    selection.registration() == null
                            || Objects.equals(requestId, selection.registration().requestId()),
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
    }

    public Response successResponse() {
        var response = new Response();
        var statuses = new ArrayList<ServerStatus>(roles.size());
        for (var role : roles) statuses.add(role.toServerStatus(requestId));
        response.setServerStatus(statuses);
        response.setSuccess(true);
        response.setCode(200);
        return response;
    }

    public PrefillEndpoint.RequestRegistration prefillRegistration() {
        var prefill = role(RoleType.PREFILL);
        return prefill == null
                ? null
                : (PrefillEndpoint.RequestRegistration) prefill.registration();
    }

    public RoleSelection role(RoleType role) {
        for (var selection : roles) {
            if (selection.role() == role
                    || role.supportsPrefill() && selection.role().supportsPrefill())
                return selection;
        }
        return null;
    }

    public EncoderEndpoint.RequestRegistration encoderRegistration() {
        var encoder = role(RoleType.ENCODER);
        return encoder == null
                ? null
                : (EncoderEndpoint.RequestRegistration) encoder.registration();
    }

    public EncoderEndpoint encoderEndpoint() {
        return (EncoderEndpoint) endpoint(RoleType.ENCODER);
    }

    public long computeTokens() {
        var prefill = role(RoleType.PREFILL);
        return prefill == null ? 0L : prefill.computeTokens();
    }

    public PrefillEndpoint prefillEndpoint() {
        return (PrefillEndpoint) endpoint(RoleType.PREFILL);
    }

    public DecodeEndpoint decodeEndpoint() {
        return (DecodeEndpoint) endpoint(RoleType.DECODE);
    }

    public DecodeEndpoint.RequestRegistration decodeRegistration() {
        var decode = role(RoleType.DECODE);
        return decode == null ? null : (DecodeEndpoint.RequestRegistration) decode.registration();
    }

    public int priority() {
        return ctx.getPriority();
    }

    public static SelectionResult<RequestRoute, CapacityKey> validateEndpoints(
            List<RoleSelection> roles, SelectionResult<RequestRoute, CapacityKey> result) {
        for (var role : roles) {
            if (!role.endpoint().acceptsNewRequests())
                return SelectionResult.retry(
                        Response.buildErrorResponse(
                                StrategyErrorType.forRole(role.role()),
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

    SelectionResult<RequestRoute, CapacityKey> validateEndpoints(
            SelectionResult<RequestRoute, CapacityKey> result) {
        return validateEndpoints(roles, result);
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

    CapacityKey capacityKey(RoleType role) {
        return role(role).capacityKey();
    }
}
