package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkState;

/**
 * Reports selected roles, wait conditions, failures and preemption completion for a request.
 * RequestContext owns live state; archived results retain the request summary and scheduler association.
 * JSON exports the public request state without scheduler or endpoint references.
 */
@lombok.Getter
@lombok.experimental.Accessors(fluent = true)
public final class RequestResult implements org.flexlb.worker.endpoint.WorkerRoute {
    private static final java.util.List<RoleType> ENDPOINT_CHECK_ORDER =
            java.util.List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.ENCODER, RoleType.VIT);

    // Only routing decisions set this field.
    private RoutingResult.Status status;
    private Response failure;
    private CapacityKey blocker;
    private java.util.Map<String, Object> diagnostics;

    @com.fasterxml.jackson.annotation.JsonProperty("originalNumericRequestId")
    private long originalNumericRequestId;

    @com.fasterxml.jackson.annotation.JsonProperty("requestPhase")
    private org.flexlb.balance.schedule.RequestPhase requestPhase;

    @com.fasterxml.jackson.annotation.JsonProperty("state")
    private State state;

    private RequestContext.EndKind endKind;
    private StrategyErrorType errorType;
    private RequestContext.ResponseKind responseKind;

    @com.fasterxml.jackson.annotation.JsonProperty("createdAtMs")
    private long createdAtMs;

    @com.fasterxml.jackson.annotation.JsonProperty("updatedAtMs")
    private long updatedAtMs;

    @com.fasterxml.jackson.annotation.JsonProperty("detail")
    private String detail = "";

    private Runnable preemptionContinuation;
    private DecodeEndpoint.RequestRegistration reservation;

    private boolean controlFailure;

    private RequestContext ctx;
    private AbstractRequestScheduler owner;

    @com.fasterxml.jackson.annotation.JsonProperty("requestId")
    private String requestId;

    private List<RoleSelection> roles = List.of();

    public RequestResult(RequestContext context, List<RoleSelection> roles) {
        this.status = RoutingResult.Status.SUCCESS;
        this.ctx = Objects.requireNonNull(context, "context");
        this.requestId = Objects.requireNonNull(context.getRequestId(), "requestId");
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
        var prefill = prefillRegistration();
        if (prefill != null) prefill.bind(this);
    }

    public RequestResult(
            String requestId,
            long originalNumericRequestId,
            org.flexlb.balance.schedule.RequestPhase requestPhase,
            State state,
            long createdAtMs,
            long updatedAtMs,
            String detail) {
        this.requestId = requestId;
        this.originalNumericRequestId = originalNumericRequestId;
        this.requestPhase = requestPhase;
        this.state = Objects.requireNonNull(state, "state");
        this.createdAtMs = createdAtMs;
        this.updatedAtMs = updatedAtMs;
        this.detail = Objects.requireNonNull(detail, "detail");
    }

    public RequestResult(
            String requestId,
            long numericId,
            org.flexlb.balance.schedule.RequestPhase requestPhase,
            State state,
            long createdAtMs,
            long updatedAtMs,
            String detail,
            AbstractRequestScheduler owner) {
        this(requestId, numericId, requestPhase, state, createdAtMs, updatedAtMs, detail);
        this.owner = owner;
    }

    public RequestResult(
            DecodeEndpoint.RequestRegistration reservation, boolean controlFailure, String detail) {
        com.google.common.base.Preconditions.checkArgument(
                reservation == null || !controlFailure,
                "committed preemption cannot be a control failure");
        this.reservation = reservation;
        this.controlFailure = controlFailure;
        this.detail = Objects.requireNonNull(detail, "detail");
    }

    public RequestResult(
            RequestContext.EndKind endKind,
            State state,
            StrategyErrorType errorType,
            String detail) {
        com.google.common.base.Preconditions.checkArgument(
                state != null && state.isTerminal(), "terminal state required");
        this.endKind = endKind;
        this.state = state;
        this.errorType = errorType;
        this.detail = detail;
    }

    public RequestResult(
            RequestContext owner, RequestContext.ResponseKind responseKind, long nowMs) {
        this.ctx = Objects.requireNonNull(owner, "owner");
        this.responseKind = Objects.requireNonNull(responseKind, "responseKind");
        this.updatedAtMs = nowMs;
    }

    private RequestResult(
            RoutingResult.Status status,
            Response failure,
            CapacityKey blocker,
            java.util.Map<String, Object> diagnostics) {
        this.status = status;
        this.failure = failure;
        this.blocker = blocker;
        this.diagnostics = diagnostics == null ? null : java.util.Map.copyOf(diagnostics);
    }

    public Response successResponse() {
        checkState(status == RoutingResult.Status.SUCCESS, "selected route required");
        var response = new Response();
        var statuses = new ArrayList<org.flexlb.worker.status.ServerStatus>(roles.size());
        for (var role : roles) statuses.add(role.toServerStatus(requestId));
        response.setServerStatus(statuses);
        response.setSuccess(true);
        response.setCode(200);
        return response;
    }

    /**
     * Exact Prefill resource record acquired during selection; presence does not imply publication.
     */
    public PrefillEndpoint.RequestRegistration prefillRegistration() {
        var prefill = (RoleSelection) role(RoleType.PREFILL);
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
        var prefill = (RoleSelection) role(RoleType.PREFILL);
        return prefill == null ? 0L : prefill.computeTokens();
    }

    public PrefillEndpoint prefillEndpoint() {
        return (PrefillEndpoint) endpoint(RoleType.PREFILL);
    }

    public DecodeEndpoint decodeEndpoint() {
        return (DecodeEndpoint) endpoint(RoleType.DECODE);
    }

    public DecodeEndpoint.RequestRegistration decodeRegistration() {
        var decode = (RoleSelection) role(RoleType.DECODE);
        return decode == null ? null : (DecodeEndpoint.RequestRegistration) decode.registration();
    }

    public boolean requestExpired(long nowMs) {
        long expiry = expiresAtMs();
        return expiry <= 0L || nowMs >= expiry;
    }

    public long expiresAtMs() {
        return ctx.getRequestExpiresAtMs();
    }

    /**
     * Normalized priority for the Worker queue index.
     */
    public int priority() {
        return ctx.getPriority();
    }

    public long responseAtMs() {
        return updatedAtMs;
    }

    public boolean committed() {
        return reservation != null;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (!(other instanceof RequestResult result)
                || ctx != null
                || result.ctx != null
                || state == null
                || result.state == null) return false;
        return originalNumericRequestId == result.originalNumericRequestId
                && createdAtMs == result.createdAtMs
                && updatedAtMs == result.updatedAtMs
                && Objects.equals(requestId, result.requestId)
                && requestPhase == result.requestPhase
                && state == result.state
                && endKind == result.endKind
                && errorType == result.errorType
                && Objects.equals(detail, result.detail);
    }

    @Override
    public int hashCode() {
        return ctx != null || state == null
                ? System.identityHashCode(this)
                : Objects.hash(
                        requestId,
                        originalNumericRequestId,
                        requestPhase,
                        state,
                        createdAtMs,
                        updatedAtMs,
                        endKind,
                        errorType,
                        detail);
    }

    public static RequestResult fromDecision(RoutingResult<?, CapacityKey> decision) {
        com.google.common.base.Preconditions.checkArgument(
                decision.status() != RoutingResult.Status.SUCCESS,
                "expected failed role selection");
        return new RequestResult(
                decision.status(), decision.failure(), decision.blocker(), decision.diagnostics());
    }

    public static RequestResult closed() {
        return new RequestResult(RoutingResult.Status.CLOSED, null, null, null);
    }

    public static RequestResult rejected(Response failure) {
        return rejected(failure, null);
    }

    public static RequestResult rejected(
            Response failure, java.util.Map<String, Object> diagnostics) {
        com.google.common.base.Preconditions.checkArgument(
                failure != null && !failure.isSuccess(), "failure response required");
        return new RequestResult(RoutingResult.Status.REJECTED, failure, null, diagnostics);
    }

    public static RequestResult blocked(CapacityKey blocker) {
        return blocked(blocker, null, null);
    }

    public static RequestResult blocked(
            CapacityKey blocker, Response failure, java.util.Map<String, Object> diagnostics) {
        com.google.common.base.Preconditions.checkArgument(
                failure == null || !failure.isSuccess(), "failure response required");
        return new RequestResult(
                RoutingResult.Status.BLOCKED,
                failure,
                Objects.requireNonNull(blocker),
                diagnostics);
    }

    /**
     * Public reporting projection; internal control facts never escape as executable commands.
     */
    public static RequestResult from(RequestContext.Snapshot snapshot) {
        if (snapshot.requestRecordForArchive() != null) {
            return snapshot.requestRecordForArchive();
        }
        if (snapshot.requestId() == null) {
            return null;
        }
        var delivery = snapshot.committedRoute();
        var cancellation = snapshot.cancellationReason();
        State phase =
                snapshot.requestEndResult() != null
                        ? snapshot.requestEndResult().state()
                        : cancellation != null
                                ? State.CANCEL_REQUESTED
                                : delivery != null ? State.ACKNOWLEDGED : State.QUEUED;
        String detail =
                snapshot.requestEndResult() != null
                        ? snapshot.requestEndResult().detail()
                        : cancellation != null
                                ? snapshot.cancellationDetail()
                                : snapshot.visibilityOverdue()
                                        ? "SUSPECTED_LOST: request not visible before its deadline"
                                        : delivery != null && snapshot.responseForFuture() != null
                                                ? "delivery acknowledged"
                                                : "queued";
        return new RequestResult(
                snapshot.requestId(),
                snapshot.originalNumericRequestId(),
                snapshot.requestPhase(),
                phase,
                snapshot.createdAtMs(),
                snapshot.updatedAtMs(),
                detail == null ? "" : detail);
    }

    public static RequestResult validateEndpoints(List<RoleSelection> roles, RequestResult result) {
        for (var role : roles) {
            if (!role.endpoint().acceptsNewRequests())
                return retry(
                        Response.buildErrorResponse(
                                org.flexlb.worker.endpoint.StrategyErrorType.forRole(role.role()),
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

    public static RequestResult retry() {
        return retry(null);
    }

    public static RequestResult retry(Response failure) {
        return new RequestResult(RoutingResult.Status.RETRY, failure, null, null);
    }

    /**
     * Validate a selection before publication, or classify a rejected admission after it.
     * An unavailable endpoint invalidates the whole selection, regardless of its role.
     * Delivered routes must retain their delivery result instead of being validated again.
     */
    RequestResult validateEndpoints(RequestResult result) {
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

    RequestResult waitingForPreemption(Runnable continuation) {
        com.google.common.base.Preconditions.checkState(
                status == RoutingResult.Status.BLOCKED, "preemption requires a capacity wait");
        var pending = new RequestResult(status, failure, blocker, diagnostics);
        pending.preemptionContinuation = Objects.requireNonNull(continuation);
        return pending;
    }

    /**
     * Query-visible request states; only cancellation, timeout, failure and completion are final.
     */
    public enum State {
        QUEUED,
        ACKNOWLEDGED,
        CANCEL_REQUESTED,
        CANCELLED,
        TIMED_OUT,
        FAILED,
        COMPLETED;

        public boolean isTerminal() {
            return this == CANCELLED || this == TIMED_OUT || this == FAILED || this == COMPLETED;
        }
    }
}
