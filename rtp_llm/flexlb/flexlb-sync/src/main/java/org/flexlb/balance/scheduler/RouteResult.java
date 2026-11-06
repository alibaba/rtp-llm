package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * Immutable ordered combination of role selections. It owns no reservations or delivery state.
 */
public final class RouteResult {
    private final String requestId;
    private final List<RoleSelection> roles;

    RouteResult(String requestId, List<RoleSelection> roles) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.roles = List.copyOf(roles);
    }

    public String requestId() {
        return requestId;
    }

    public List<RoleSelection> roles() {
        return roles;
    }

    public RoleSelection role(RoleType role) {
        for (var selection : roles) {
            if (selection.role() == role
                    || role.supportsPrefill() && selection.role().supportsPrefill())
                return selection;
        }
        return null;
    }

    WorkerEndpoint endpoint(RoleType role) {
        var selected = role(role);
        return selected == null ? null : selected.endpoint();
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

    /**
     * Thread-confined selection scope; discarded after the immutable result is prepared.
     */
    public static final class Builder {
        private boolean acceptingSelections = true;

        private final RequestContext ctx;
        private ArrayList<RoleSelection> selections;

        public Builder(RequestContext context) {
            ctx = Objects.requireNonNull(context, "context");
        }

        RequestContext ctx() {
            return ctx;
        }

        private String requestId() {
            return ctx.data().getRequestId();
        }

        public ServerStatus add(
                WorkerEndpoint selectedEndpoint, ServerStatus status, long workMs, long version) {
            try {
                checkState(acceptingSelections, "route selection is not building");
                ServerStatus metadata = validateSelection(selectedEndpoint, status);
                RoleType role = metadata.getRole();
                checkState(
                        selection(role) == null,
                        "request requires at most one assignment per role");
                var selection = RoleSelection.capture(selectedEndpoint, metadata, workMs, version);
                if (selections == null) selections = new ArrayList<>(3);
                selections.add(selection);
                return status;
            } catch (RuntimeException | Error failure) {
                acceptingSelections = false;
                throw failure;
            }
        }

        private ServerStatus validateSelection(
                WorkerEndpoint selectedEndpoint, ServerStatus status) {
            checkState(selectedEndpoint != null, "selected endpoint is required");
            ServerStatus frozen = Objects.requireNonNull(status, "worker metadata");
            checkArgument(frozen.isSuccess(), "route requires successful Worker metadata");
            checkArgument(
                    frozen.getRole() != null && frozen.getRole() != RoleType.FRONTEND,
                    "unsupported Worker role");
            checkState(
                    java.util.Objects.equals(frozen.getRequestId(), requestId()),
                    "selected Worker belongs to another request");
            WorkerEndpoint endpoint = selectedEndpoint;
            checkArgument(
                    Objects.equals(frozen.getServerIp(), endpoint.getIp())
                            && frozen.getHttpPort() == endpoint.getHttpPort()
                            && frozen.getRoutingEngineIndex()
                                    == endpoint.getStatus().getEngineIndex(),
                    "selection metadata does not match selected endpoint address");
            return frozen;
        }

        /**
         * Freeze the selected roles once. Selection itself owns no resources to release.
         */
        RouteResult prepare() {
            checkState(acceptingSelections, "route selection is not building");
            acceptingSelections = false;
            checkState(
                    selection(RoleType.PREFILL) != null || selection(RoleType.ENCODER) != null,
                    "one Prefill or Encoder route required");
            return new RouteResult(requestId(), selections);
        }

        RoleSelection selection(RoleType role) {
            if (selections != null)
                for (var selection : selections) {
                    if (selection.role() == role
                            || role.supportsPrefill() && selection.role().supportsPrefill())
                        return selection;
                }
            return null;
        }

        <T> PlacementResult<T, PlacementKey> validateEndpoints(
                PlacementResult<T, PlacementKey> result) {
            var roles = selections;
            if (roles == null) return result;
            for (var assignment : roles) {
                WorkerEndpoint endpoint = assignment.endpoint();
                if (endpoint.acceptsNewRequests()) continue;
                acceptingSelections = false;
                return PlacementResult.endpointChanged(
                        assignment.placementKey(),
                        PlacementResult.success(new RouteResult(requestId(), roles)));
            }
            return result;
        }
    }
}
