package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillRequestQueue;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 * One route execution identity, retained by queues and callbacks until its resources are released.
 * Selection facts live only in RouteResult; this object owns reservation and delivery handles.
 */
public final class RouteExecution implements GroupPlanner.Input, PrefillRequestQueue.QueueItem {

    private static final java.util.List<RoleType> ENDPOINT_CHECK_ORDER =
            java.util.List.of(RoleType.PREFILL, RoleType.DECODE, RoleType.ENCODER, RoleType.VIT);
    private final RequestContext ctx;
    private final RouteResult result;
    private boolean encoderCommitted;
    private DecodeResources.RequestReservation decodeReservation;
    private PrefillEndpoint.RequestRecord prefillReservation;
    private volatile Delivery deliveryPreparation;

    synchronized void attachDeliveryPreparation(Delivery preparation) {
        checkState(
                deliveryPreparation == null || deliveryPreparation.preparationReleased(),
                "route already has delivery preparation");
        deliveryPreparation = preparation;
    }

    synchronized void clearDeliveryPreparation(Delivery preparation) {
        if (deliveryPreparation == preparation) deliveryPreparation = null;
    }

    void afterDeliveryPreparation(java.util.function.Consumer<Throwable> completion) {
        var preparation = deliveryPreparation;
        if (preparation == null) completion.accept(null);
        else preparation.afterPreparationReleased(completion);
    }

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

    /**
     * Construction and capacity attachment finish before this identity is published to Context.
     */
    public RouteExecution(RequestContext context, RouteResult result) {
        this.ctx = Objects.requireNonNull(context, "context");
        this.result = Objects.requireNonNull(result, "result");
        checkArgument(
                Objects.equals(context.data().getRequestId(), result.requestId()),
                "route belongs to another request");
    }

    static PlacementResult<RouteExecution, PlacementKey> fromSelection(
            RequestContext context, PlacementResult<RouteResult, PlacementKey> selected) {
        return selected.status() == PlacementResult.Status.SUCCESS
                ? new PlacementResult<>(
                        selected.status(),
                        new RouteExecution(context, selected.value()),
                        null,
                        null,
                        selected.diagnostics(),
                        null)
                : selected.asFailure();
    }

    /**
     * Attach the acquired capacity once, before queue or delivery publication.
     */
    RouteExecution attachReservation(DecodeResources.RequestReservation reservation) {
        if (decodeReservation != null) {
            checkState(
                    Objects.equals(decodeReservation, reservation),
                    "route capacity identity already assigned");
            return this;
        }
        var inputs = Objects.requireNonNull(ctx.data().getRequest(), "request inputs");
        Objects.requireNonNull(ctx.data().getSchedulingMetadata(), "scheduling metadata");
        checkArgument(
                reservation == null
                        || java.util.Objects.equals(reservation.requestId(), inputs.getRequestId()),
                "Decode reservation belongs to another request");
        decodeReservation = reservation;
        return this;
    }

    WorkerEndpoint blockedEndpointIfCurrent(PlacementKey blocker) {
        boolean isDecode = blocker.role() == RoleType.DECODE;
        checkArgument(
                (isDecode ? decodeEndpoint() : prefillEndpoint()) != null
                        && blocker.equals(placementKey(blocker.role())),
                "blocker does not belong to this Worker assignment");
        WorkerEndpoint endpoint = isDecode ? decodeEndpoint() : prefillEndpoint();
        long version =
                endpoint instanceof PrefillEndpoint worker
                        ? worker.progress().ownershipVersion()
                        : ((DecodeEndpoint) endpoint).placementVersion();
        return version
                        == (isDecode
                                ? ((RoleSelection.Decode) selection(RoleType.DECODE))
                                        .resourceVersion()
                                : ((RoleSelection.Prefill) selection(RoleType.PREFILL))
                                        .resourceVersion())
                ? endpoint
                : null;
    }

    /**
     * Validate a selection before publication, or classify a rejected admission after it.
     * An unavailable endpoint invalidates the whole selection, regardless of its role.
     * Delivered routes must retain their delivery result instead of being validated again.
     */
    <T> PlacementResult<T, PlacementKey> validateEndpoints(
            PlacementResult<T, PlacementKey> result) {
        for (var role : this.result.roles()) {
            if (!role.endpoint().acceptsNewRequests()) {
                return PlacementResult.endpointChanged(
                        role.placementKey(), PlacementResult.success(this));
            }
        }
        return result;
    }

    boolean selectedEndpointsAvailable() {
        return firstUnavailableEndpoint() == null;
    }

    boolean usesEndpoint(WorkerEndpoint endpoint) {
        return endpoint == prefillEndpoint()
                || endpoint == decodeEndpoint()
                || endpoint == encoderEndpoint()
                || endpoint == endpoint(RoleType.VIT);
    }

    WorkerEndpoint firstUnavailableEndpoint() {
        for (RoleType role : ENDPOINT_CHECK_ORDER) {
            var endpoint = endpoint(role);
            if (!availableEndpoint(endpoint)) return endpoint;
        }
        return null;
    }

    private static boolean availableEndpoint(WorkerEndpoint endpoint) {
        return endpoint == null || endpoint.acceptsNewRequests();
    }

    private RoleSelection selection(RoleType role) {
        return result.role(role);
    }

    private WorkerEndpoint endpoint(RoleType role) {
        var selected = selection(role);
        return selected == null ? null : selected.endpoint();
    }

    public RouteResult result() {
        return result;
    }

    PlacementKey placementKey(RoleType role) {
        return selection(role).placementKey();
    }

    public EncoderEndpoint encoderEndpoint() {
        return (EncoderEndpoint) endpoint(RoleType.ENCODER);
    }

    boolean encoderCommitted() {
        return encoderCommitted;
    }

    void markEncoderCommitted() {
        encoderCommitted = true;
    }

    public long prefillWorkMs() {
        var prefill = (RoleSelection.Prefill) selection(RoleType.PREFILL);
        return prefill == null ? 0L : prefill.workMs();
    }

    public RequestContext ctx() {
        return ctx;
    }

    public Response successResponse(boolean enqueuedByMaster) {
        return result().successResponse(enqueuedByMaster);
    }

    public ServerStatus prefill() {
        var selected = selection(RoleType.PREFILL);
        return selected == null ? null : selected.toServerStatus(requestId());
    }

    public ServerStatus decode() {
        var selected = selection(RoleType.DECODE);
        return selected == null ? null : selected.toServerStatus(requestId());
    }

    public long prefillDpRank() {
        return selection(RoleType.PREFILL).dpRank();
    }

    public PrefillEndpoint prefillEndpoint() {
        return (PrefillEndpoint) endpoint(RoleType.PREFILL);
    }

    public DecodeEndpoint decodeEndpoint() {
        return (DecodeEndpoint) endpoint(RoleType.DECODE);
    }

    /**
     * Exact Prefill resource record; its presence records the physical handoff before callbacks.
     */
    public PrefillEndpoint.RequestRecord prefillReservation() {
        return prefillReservation;
    }

    public void attachPrefillReservation(PrefillEndpoint.RequestRecord reservation) {
        checkArgument(reservation.belongsTo(this), "Prefill reservation belongs to another route");
        prefillReservation = reservation;
    }

    void clearPrefillReservation(PrefillEndpoint.RequestRecord reservation) {
        checkState(
                prefillReservation == reservation,
                "rollback requires the exact Prefill reservation");
        prefillReservation = null;
    }

    public DecodeResources.RequestReservation decodeReservation() {
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

    // -- derived accessors --

    public String requestId() {
        return ctx.data().getRequestId();
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
        return selection(RoleType.PREFILL).hitCacheTokens();
    }
}
