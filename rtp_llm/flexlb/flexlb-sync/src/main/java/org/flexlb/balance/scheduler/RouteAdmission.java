package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.DecodeEndpointSnapshot;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.PriorityNormalizer;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;

/** Exact selected route and its provisional resources, shared by both scheduling modes. */
public final class RouteAdmission implements AutoCloseable {
    private enum Ownership { PROVISIONAL, COMMITTED, CLOSED }

    private final long requestId;
    private final Response response;
    private final Selection selection;
    private DecodeBinding decodeBinding;
    private WorkerEndpoint blockedEndpoint;
    private Ownership ownership = Ownership.PROVISIONAL;

    private RouteAdmission(long requestId, Response response, Selection selection,
                           DecodeBinding decodeRequest) {
        this.requestId = requestId;
        this.response = Objects.requireNonNull(response, "response");
        this.selection = Objects.requireNonNull(selection, "selection");
        this.decodeBinding = Objects.requireNonNull(decodeRequest, "decodeRequest");
        if (decodeRequest.requestId() != requestId) {
            throw new IllegalArgumentException("Decode admission belongs to another request");
        }
    }

    static RouteAdmission prepare(
            BalanceContext context,
            List<SelectedRole> selectedRoles,
            Response response,
            DecodeBinding decodeRequest) {
        long requestId = context.getRequestId();

        PrefillEndpoint prefillEndpoint = null;
        WorkerEndpoint.GenerationPin prefillPin = null;
        ServerStatus prefillStatus = null;
        long prefillPlacementVersion = 0L;
        long prefillWorkMs = 0L;
        DecodeEndpoint decodeEndpoint = null;
        WorkerEndpoint.GenerationPin decodePin = null;
        ServerStatus decodeStatus = null;
        long decodePlacementVersion = 0L;

        try {
            for (SelectedRole selected : selectedRoles) {
                ServerStatus status = selected.serverStatus();
                if (status.getRequestId() != requestId) {
                    throw new IllegalStateException(
                            "selected role belongs to another request");
                }
                RoleType role = status.getRole();
                long placementVersion = selected.placementVersion();
                WorkerEndpoint.GenerationPin pin =
                        selected.takeGenerationPin();
                WorkerEndpoint endpoint = pin.endpoint();
                if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
                    if (prefillPin != null
                            || !(endpoint instanceof PrefillEndpoint prefill)) {
                        pin.close();
                        throw new IllegalStateException(
                                "route requires one exact Prefill selection");
                    }
                    prefillEndpoint = prefill;
                    prefillPin = pin;
                    prefillStatus = status;
                    prefillPlacementVersion = placementVersion;
                    prefillWorkMs = selected.prefillWorkMs();
                    continue;
                }
                if (role == RoleType.DECODE) {
                    if (decodePin != null
                            || !(endpoint instanceof DecodeEndpoint decode)) {
                        pin.close();
                        throw new IllegalStateException(
                                "route requires at most one exact Decode selection");
                    }
                    decodeEndpoint = decode;
                    decodePin = pin;
                    decodeStatus = status;
                    decodePlacementVersion = placementVersion;
                    continue;
                }
                // Stateless roles need no owner after their response metadata
                // has been frozen.
                pin.close();
            }
            if (prefillPin == null || prefillEndpoint == null
                    || prefillStatus == null) {
                throw new IllegalStateException(
                        "route has no Prefill endpoint generation");
            }
            return new RouteAdmission(
                    requestId,
                    response,
                    new Selection(
                            prefillEndpoint,
                            prefillPin,
                            prefillStatus,
                            prefillPlacementVersion,
                            prefillWorkMs,
                            decodePin,
                            decodePlacementVersion),
                    decodeRequest.bind(decodeStatus, decodeEndpoint, null));
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                throw failure;
            }
        }
    }

    static RouteAdmission prepare(BalanceContext context, List<SelectedRole> selections, Response response) {
        return prepare(context, selections, response, DecodeBinding.capture(context));
    }

    WorkerEndpoint blockedEndpoint() { return blockedEndpoint; }

    /**
     * Whether a retry must wait for more capacity at this endpoint, using the
     * current request's admission policy. Unknown capacity or a possible
     * preemption returns false so normal routing can decide. Before parking,
     * the caller must recheck that no newer capacity notification has arrived.
     */
    static boolean mustWaitForCapacity(BalanceContext context, WorkerEndpoint endpoint) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                return false;
            }
            if (endpoint instanceof PrefillEndpoint prefill) {
                return !prefill.canAcceptRequest();
            }
            if (!(endpoint instanceof DecodeEndpoint decode)) {
                return false;
            }
            DecodeBinding request = DecodeBinding.capture(context);
            if (request.mode() != ScheduledRequest.DecodeMode.PREEMPT_AT_PLACEMENT) {
                return false;
            }
            DecodeEndpointSnapshot snapshot = DecodeEndpointSnapshot.capture(decode, request.capacity());
            if (request.capacity().evaluate(snapshot.usage(),
                    request.hardKvTokens(), request.expectedKvTokens()).fits()) {
                return false;
            }
            // Exclude only cases with no possible lower-priority victim. Actual
            // victim eligibility and feasibility remain owned by EvictionManager.
            return Stream.of(snapshot.reserved(), snapshot.accepted(), snapshot.running())
                    .flatMap(List::stream)
                    .noneMatch(victim -> PriorityNormalizer.hasPriority(victim.priority())
                            && victim.priority() < request.priority());
        }
    }

    boolean blockedEndpointChanged() {
        requireProvisional();
        if (blockedEndpoint == prefillEndpoint()) {
            return prefillPlacementVersion() != prefillEndpoint().placementVersion();
        }
        return blockedEndpoint == decodeEndpoint() && decodeEndpoint() != null
                && decodePlacementVersion() != decodeEndpoint().placementVersion();
    }

    public PlacementResult<ScheduledRequest, PlacementKey> tryEnqueue(
            BalanceContext context, CompletableFuture<Response> future, RequestRegistry lifecycle) {
        requireProvisional();
        blockedEndpoint = null;
        if (!reserveDecode()) {
            blockedEndpoint = decodeEndpoint();
            return PlacementResult.blocked(decodePlacementKey());
        }
        ScheduledRequest item = createScheduledRequest(context, future, System.currentTimeMillis());
        context.setRouteSubmittedNanos(System.nanoTime());
        PlacementResult.Status result = lifecycle.commitRoute(item,
                () -> prefillEndpoint().offerPinned(prefillPin(), item));
        return switch (result) {
            case SUCCESS -> {
                markCommitted();
                yield PlacementResult.success(item);
            }
            case BLOCKED -> {
                blockedEndpoint = prefillEndpoint();
                yield PlacementResult.blocked(PlacementKey.exact(prefillStatus().getRole(),
                        prefillStatus().getGroup(), prefillEndpoint().ipPort()));
            }
            case CLOSED -> PlacementResult.closed();
            case REJECTED -> throw new IllegalStateException("route commit cannot reject with a response");
        };
    }

    public boolean commitQueuedRequest(RequestRegistry lifecycle, ScheduledRequest item) {
        requireProvisional();
        if (lifecycle.commitRoute(item, () -> prefillEndpoint().offerPinned(prefillPin(), item))
                != PlacementResult.Status.SUCCESS) { return false; }
        markCommitted();
        return true;
    }

    Response response() { return response; }
    public PrefillEndpoint prefillEndpoint() { return selection.prefillEndpoint(); }
    public DecodeEndpoint decodeEndpoint() { return decodeBinding.endpoint(); }
    private WorkerEndpoint.GenerationPin prefillPin() { return selection.prefillPin(); }
    private WorkerEndpoint.GenerationPin decodePin() { return selection.decodePin(); }
    private ServerStatus prefillStatus() { return selection.prefillStatus(); }
    private ServerStatus decodeStatus() { return decodeBinding.status(); }
    private long prefillPlacementVersion() { return selection.prefillPlacementVersion(); }
    private long decodePlacementVersion() { return selection.decodePlacementVersion(); }
    public DecodeBinding decodeBinding() { return decodeBinding; }

    boolean reserveDecode() {
        requireProvisional();
        if (decodeEndpoint() == null || decodeBinding.reservation() != null) { return true; }
        DecodeEndpoint.ReservationHandle reservation = switch (decodeBinding.mode()) {
            case IMMEDIATE, WAIT_AT_DISPATCH -> decodeEndpoint().tryReservePlacementPinned(
                    decodePin(), requestId, decodeBinding.hardKvTokens(), decodeBinding.expectedKvTokens(),
                    decodeBinding.priority());
            case PREEMPT_AT_PLACEMENT -> decodeEndpoint().tryReservePlacementPinned(
                    decodePin(), requestId, decodeBinding.hardKvTokens(), decodeBinding.expectedKvTokens(),
                    decodeBinding.priority(), decodeBinding.capacity());
        };
        if (reservation == null) { return false; }
        try {
            decodeBinding = decodeBinding.bind(decodeStatus(), decodeEndpoint(), reservation);
            return true;
        } catch (RuntimeException | Error failure) {
            decodeEndpoint().releaseReservationExact(reservation);
            throw failure;
        }
    }

    public boolean adoptDecodeReservation(DecodeEndpoint endpoint, DecodeEndpoint.ReservationHandle reservation) {
        requireProvisional();
        if (endpoint == null || endpoint != decodeEndpoint() || reservation == null || reservation.requestId() != requestId
                || decodeBinding.reservation() != null) {
            if (endpoint != null && reservation != null) { endpoint.releaseReservationExact(reservation); }
            return false;
        }
        try {
            endpoint.requirePinnedGeneration(decodePin());
            if (!endpoint.markQueuedExact(decodePin(), reservation)) {
                endpoint.releaseReservationExact(reservation);
                return false;
            }
            decodeBinding = decodeBinding.bind(decodeStatus(), decodeEndpoint(), reservation);
            return true;
        } catch (RuntimeException | Error failure) {
            endpoint.releaseReservationExact(reservation);
            throw failure;
        }
    }

    public ScheduledRequest createScheduledRequest(BalanceContext context, CompletableFuture<Response> future, long enqueuedAtMs) {
        requireProvisional();
        if (context.getRequestId() != requestId) {
            throw new IllegalArgumentException("admission cannot build another request");
        }
        return new ScheduledRequest(context, future, response,
                RequestRegistry.copyOf(prefillStatus()), RequestRegistry.copyOf(decodeStatus()),
                prefillEndpoint(), decodeEndpoint(), decodeBinding.reservation(), enqueuedAtMs, decodeBinding);
    }

    /** The caller has registered the same canonical request before selecting this route. */
    record RouteDelivery(RequestRegistry.DeliveryClaim claim, WorkSnapshot precedingWork, long unstartedWorkMs) {
        RouteDelivery {
            Objects.requireNonNull(claim, "claim");
            Objects.requireNonNull(precedingWork, "precedingWork");
        }
    }

    PlacementResult<RouteDelivery, PlacementKey> tryCommitDirectRoute(
            BalanceContext context, RequestRegistry lifecycle) {
        requireProvisional();
        try (AdmissionMutation mutation = lifecycle.claimAdmissionMutation(requestId, context.getFuture())) {
            if (mutation == null) { return PlacementResult.closed(); }
            if (!reserveDecode()) {
                return PlacementResult.blocked(decodePlacementKey());
            }
            ScheduledRequest item = createScheduledRequest(context, context.getFuture(), context.getEnqueueTime());
            CapacityBoundary.Attempt<PrefillAdmissionResources.Member> attempt =
                    PrefillAdmissionResources.prepareMember(item);
            if (!attempt.accepted()) {
                return switch (attempt.boundary().status()) {
                    case UNAVAILABLE -> PlacementResult.blocked(decodePlacementKey());
                    case OWNERSHIP_LOST -> PlacementResult.rejected(
                            Response.error(StrategyErrorType.SCHEDULER_PLAN_CONFLICT));
                    case FAILED -> throw new IllegalStateException("route admission failed", attempt.boundary().cause());
                };
            }
            PrefillAdmissionResources.Member member = attempt.value();
            try {
                return commitImmediateRoute(lifecycle, item, member);
            } finally {
                Throwable cleanup = PrefillAdmissionResources.rollbackMember(member, null);
                if (cleanup != null) { PrefillAdmissionResources.throwRollbackFailure(cleanup); }
            }
        }
    }

    private PlacementResult<RouteDelivery, PlacementKey> commitImmediateRoute(
            RequestRegistry lifecycle, ScheduledRequest item,
            PrefillAdmissionResources.Member member) {
        var reservationAttempt = new PrefillReservationAttempt(item);
        try (PrefillEndpoint.RouteCommitAdmission routeCommit = prefillEndpoint().tryBeginRouteCommitAdmission()) {
            if (routeCommit == null) {
                return PlacementResult.rejected(Response.error(StrategyErrorType.DISPATCH_FAILED));
            }
            try (var admission = PrefillAdmissionResources.createCommittedOwner(List.of(member))) {
                PlacementResult.Status result = lifecycle.commitRoute(item, reservationAttempt);
                if (result != PlacementResult.Status.SUCCESS) {
                    return result == PlacementResult.Status.CLOSED ? PlacementResult.closed()
                            : PlacementResult.rejected(Response.error(reservationAttempt.failure()));
                }
                var handoff = routeCommit.commit(List.of(item), List.of(reservationAttempt.reservation()));
                admission.bindPrefillHandoff(handoff);
                var claim = lifecycle.tryClaimRouteDelivery(item, () -> admission.transferToEndpoint(item));
                if (claim == null) { return PlacementResult.closed(); }
                markCommitted();
                return PlacementResult.success(new RouteDelivery(claim, handoff.precedingWork(), selection.prefillWorkMs()));
            }
        } finally {
            if (reservationAttempt.reservation() != null) { reservationAttempt.reservation().close(); }
        }
    }

    private PlacementKey decodePlacementKey() {
        return PlacementKey.exact(decodeStatus().getRole(), decodeStatus().getGroup(), decodeEndpoint().ipPort());
    }

    /** Holds acquisition output without mixing a failed publication with capacity pressure. */
    private final class PrefillReservationAttempt implements BooleanSupplier {
        private final ScheduledRequest item;
        private PrefillState.ReservationResult<PrefillState.RouteReservation> result;

        private PrefillReservationAttempt(ScheduledRequest item) {
            this.item = item;
        }

        @Override
        public boolean getAsBoolean() {
            result = prefillEndpoint().reserveUnqueuedRoute(prefillPin(), item,
                    selection.prefillWorkMs());
            return result.status() == PrefillState.CapacityStatus.ACQUIRED;
        }

        private PrefillState.RouteReservation reservation() {
            return result == null ? null : result.reservation();
        }

        private StrategyErrorType failure() {
            if (result == null) { return StrategyErrorType.REQUEST_CANCELLED; }
            return switch (result.status()) {
                case CAPACITY_FULL -> StrategyErrorType.RESOURCE_EXHAUSTED;
                case ENDPOINT_RETIRED -> StrategyErrorType.DISPATCH_FAILED;
                case REQUEST_ALREADY_RESERVED, REQUEST_NOT_ACTIVE, BATCH_ID_ALREADY_RESERVED ->
                        StrategyErrorType.SCHEDULER_PLAN_CONFLICT;
                case ACQUIRED -> throw new IllegalStateException("successful admission cannot be rejected");
            };
        }
    }

    private void markCommitted() {
        requireProvisional();
        ownership = Ownership.COMMITTED;
        closePins();
    }

    private void requireProvisional() {
        if (ownership != Ownership.PROVISIONAL) { throw new IllegalStateException("route admission already resolved"); }
    }

    @Override
    public void close() {
        if (ownership != Ownership.PROVISIONAL) { return; }
        ownership = Ownership.CLOSED;
        try {
            if (decodeBinding.reservation() != null) { decodeEndpoint().releaseReservationExact(decodeBinding.reservation()); }
        } finally {
            closePins();
        }
    }

    private void closePins() {
        try (var prefill = prefillPin(); var decode = decodePin()) { }
    }

    private record Selection(PrefillEndpoint prefillEndpoint, WorkerEndpoint.GenerationPin prefillPin,
                             ServerStatus prefillStatus, long prefillPlacementVersion, long prefillWorkMs,
                             WorkerEndpoint.GenerationPin decodePin, long decodePlacementVersion) { }
}
