package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.BooleanSupplier;

/**
 * Sole owner of a selected queue route before ACTIVE publication.
 *
 * <p>Ownership is transferred once from the planner to the commit sequencer;
 * the object is never accessed concurrently. It owns both
 * endpoint-generation pins. Exact Decode capacity is acquired only by the
 * ordered publication transaction; planning never mutates endpoint capacity.
 * A successful commit transfers ownership to the canonical RequestSlot /
 * ScheduledRequest, while closing rolls back local pin ownership.</p>
 */
public final class QueueRouteAdmission implements AutoCloseable {

    private final long requestId;
    private final Response response;
    private final PrefillEndpoint selectedPrefillEndpoint;
    private final DecodeEndpoint selectedDecodeEndpoint;
    private OwnedRoute ownedRoute;
    /** Exact endpoint which rejected the last publication attempt. */
    private WorkerEndpoint blockedEndpoint;

    private QueueRouteAdmission(
            long requestId,
            Response response,
            OwnedRoute ownedRoute) {
        this.requestId = requestId;
        this.response = Objects.requireNonNull(response, "response");
        this.ownedRoute = Objects.requireNonNull(ownedRoute, "ownedRoute");
        this.selectedPrefillEndpoint = ownedRoute.prefillEndpoint();
        this.selectedDecodeEndpoint = ownedRoute.decodeEndpoint();
    }

    static QueueRouteAdmission prepare(
            BalanceContext context,
            List<SelectedRole> selectedRoles,
            Response response) {
        long requestId = context.getRequestId();

        PrefillEndpoint prefillEndpoint = null;
        WorkerEndpoint.GenerationPin prefillPin = null;
        ServerStatus prefillStatus = null;
        long prefillPlacementVersion = 0L;
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
                                "queue route requires one exact Prefill selection");
                    }
                    prefillEndpoint = prefill;
                    prefillPin = pin;
                    prefillStatus = status;
                    prefillPlacementVersion = placementVersion;
                    continue;
                }
                if (role == RoleType.DECODE) {
                    if (decodePin != null
                            || !(endpoint instanceof DecodeEndpoint decode)) {
                        pin.close();
                        throw new IllegalStateException(
                                "queue route requires at most one exact Decode selection");
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
                        "queue route has no Prefill endpoint generation");
            }
            return new QueueRouteAdmission(
                    requestId,
                    response,
                    new OwnedRoute(
                            prefillEndpoint,
                            prefillPin,
                            prefillStatus,
                            prefillPlacementVersion,
                            decodeEndpoint,
                            decodePin,
                            null,
                            decodeStatus,
                            decodePlacementVersion));
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                throw failure;
            }
        }
    }

    Response response() {
        return response;
    }

    /**
     * Returns the exact endpoint whose local capacity rejected publication.
     * A queue coordinator may bypass this request only for a route which does
     * not use this endpoint. A null value means the selector itself had no
     * concrete endpoint and therefore cannot safely be bypassed.
     */
    WorkerEndpoint blockedEndpoint() {
        return blockedEndpoint;
    }

    /** Whether the exact capacity miss invalidated the planning snapshot. */
    boolean blockedSelectionBecameStale() {
        OwnedRoute route = requireOwned();
        if (blockedEndpoint == route.prefillEndpoint()) {
            return route.prefillPlacementVersion()
                    != route.prefillEndpoint().placementVersion();
        }
        if (blockedEndpoint == route.decodeEndpoint()) {
            return route.decodePlacementVersion()
                    != route.decodeEndpoint().placementVersion();
        }
        return false;
    }

    public WorkerBatcher.QueueSnapshot capturePrefillQueueSnapshot() {
        return requireOwned().prefillEndpoint().captureQueueSnapshot();
    }

    /** Exact Decode selected by the only routing decision, if this model has one. */
    DecodeEndpoint selectedDecodeEndpoint() {
        return selectedDecodeEndpoint;
    }

    PrefillEndpoint selectedPrefillEndpoint() {
        return selectedPrefillEndpoint;
    }

    /**
     * Consume the reservation created by exact-endpoint preemption without
     * selecting either role again. Failure releases the supplied reservation.
     */
    public boolean adoptDecodeReservation(
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        OwnedRoute route = requireOwned();
        if (endpoint == null || reservation == null
                || endpoint != route.decodeEndpoint()
                || reservation.requestId() != requestId
                || route.decodeReservation() != null) {
            if (endpoint != null && reservation != null) {
                endpoint.releaseReservationExact(reservation);
            }
            return false;
        }
        try {
            endpoint.requirePinnedGeneration(route.decodePin());
            if (!endpoint.markQueuedExact(route.decodePin(), reservation)) {
                endpoint.releaseReservationExact(reservation);
                return false;
            }
            ownedRoute = route.withDecodeReservation(reservation);
            return true;
        } catch (RuntimeException | Error failure) {
            endpoint.releaseReservationExact(reservation);
            throw failure;
        }
    }

    public ScheduledRequest buildItem(
            BalanceContext context,
            CompletableFuture<Response> future,
            long enqueuedAtMs) {
        if (context.getRequestId() != requestId) {
            throw new IllegalArgumentException(
                    "queue admission cannot build another request");
        }
        OwnedRoute route = requireOwned();
        return new ScheduledRequest(
                context,
                future,
                response,
                RequestRegistry.copyOf(route.prefillStatus()),
                RequestRegistry.copyOf(route.decodeStatus()),
                route.prefillEndpoint(),
                route.decodeEndpoint(),
                route.decodeReservation(),
                enqueuedAtMs);
    }

    /** Acquire exact Decode and Prefill capacity, then publish once. */
    public PlacementResult<ScheduledRequest, PlacementKey> tryPublish(
            BalanceContext context,
            CompletableFuture<Response> future,
            RequestRegistry lifecycle) {
        OwnedRoute route = requireOwned();
        blockedEndpoint = null;
        if (!tryReserveDecode(context, route)) {
            return PlacementResult.blocked(
                    placementKey(
                            route.decodeStatus(), route.decodeEndpoint()));
        }
        route = requireOwned();
        ScheduledRequest item = buildItem(
                context, future, System.currentTimeMillis());
        context.setRouteSubmittedNanos(System.nanoTime());
        OwnedRoute exact = route;
        PlacementResult.Status committed = commitPublication(
                exact,
                lifecycle,
                item,
                () -> exact.prefillEndpoint().offerPinned(
                        exact.prefillPin(), item));
        return switch (committed) {
            case SUCCESS -> PlacementResult.success(item);
            case BLOCKED -> {
                blockedEndpoint = exact.prefillEndpoint();
                yield PlacementResult.blocked(
                        placementKey(
                                exact.prefillStatus(),
                                exact.prefillEndpoint()));
            }
            case LIMIT_REACHED -> PlacementResult.limitReached();
            case CLOSED -> PlacementResult.closed();
            case REJECTED -> throw new IllegalStateException(
                    "route commit cannot reject with a response");
        };
    }

    private boolean tryReserveDecode(
            BalanceContext context, OwnedRoute route) {
        if (route.decodeEndpoint() == null
                || route.decodeReservation() != null) {
            return true;
        }
        long sequenceLength = Math.max(
                0L, context.getRequest().getSeqLen());
        long expectedKv = context.getConfig().decodeKvReservationTokens(
                sequenceLength,
                context.getRequest().getMaxNewTokens(),
                route.decodeEndpoint().realKvTotal());
        DecodeEndpoint.ReservationHandle reservation;
        if (context.getConfig().defersDecodeCapacityUntilDispatch()) {
            // This is a soft queued hold. The WorkerBatcher acquires the exact
            // concurrency/KV permit immediately before engine delivery, so a
            // long Prefill backlog cannot make an idle Decode pool appear full.
            reservation = route.decodeEndpoint().tryReserveQueuedPinned(
                    route.decodePin(),
                    requestId,
                    sequenceLength,
                    expectedKv,
                    context.getPriority());
        } else {
            // Preemptive ordering needs the typed placement miss to enter its
            // victim-planning path before the request is published to a queue.
            reservation = route.decodeEndpoint().tryReserveQueuedPinned(
                    route.decodePin(),
                    requestId,
                    sequenceLength,
                    expectedKv,
                    context.getPriority(),
                    decodeCapacity(context));
        }
        if (reservation == null) {
            blockedEndpoint = route.decodeEndpoint();
            return false;
        }
        blockedEndpoint = null;
        ownedRoute = route.withDecodeReservation(reservation);
        return true;
    }

    private static DecodeEndpoint.AdmissionCapacity decodeCapacity(
            BalanceContext context) {
        var availability = context.getConfig().getRouter().getRoles()
                .getDecode().getAvailability();
        Long maximumRequests = availability.getMaxEngineRequests();
        return new DecodeEndpoint.AdmissionCapacity(
                maximumRequests == null ? 0L : maximumRequests,
                availability.getMaxKvUsagePercent());
    }

    private static PlacementKey placementKey(
            ServerStatus status,
            WorkerEndpoint endpoint) {
        return PlacementKey.exact(
                status.getRole(), status.getGroup(), endpoint.ipPort());
    }

    /**
     * Atomically bind the exact item and publish it ACTIVE.  Pins are released
     * only after the registrar has dropped the slot monitor and WorkerBatcher
     * has dropped its queue lock.
     */
    public boolean commitTo(
            RequestRegistry lifecycle,
            ScheduledRequest item) {
        OwnedRoute route = requireOwned();
        return commitPublication(
                route,
                lifecycle,
                item,
                () -> tryOfferPinned(route, item))
                == PlacementResult.Status.SUCCESS;
    }

    /** Execute one admission-owned endpoint mutation and ACTIVE publication. */
    private PlacementResult.Status commitPublication(
            OwnedRoute route,
            RequestRegistry lifecycle,
            ScheduledRequest item,
            BooleanSupplier activePublication) {
        if (requireOwned() != route) {
            throw new IllegalStateException(
                    "queue admission ownership changed before publication");
        }
        var lifecycleConfig = item.ctx().getConfig()
                .queueScheduler().getLifecycle();
        PlacementResult.Status committed =
                lifecycle.commitRoute(
                item,
                lifecycleConfig.getMaxDeliveredNotAcceptedRequestsGlobal(),
                lifecycleConfig.getDeliveredNotAcceptedTimeoutMs(),
                activePublication);
        if (committed
                == PlacementResult.Status.SUCCESS) {
            finishCommitted(route);
        }
        return committed;
    }

    /** Result of one commit-time route reservation and exact queue replacement. */
    public record QueueReplacementCommit(
            WorkerBatcher.QueueReplacementStatus status,
            ScheduledRequest item) {

        public QueueReplacementCommit {
            Objects.requireNonNull(status, "status");
            if ((status == WorkerBatcher.QueueReplacementStatus.SUCCESS)
                    != (item != null)) {
                throw new IllegalArgumentException(
                        "only a successful replacement owns the committed item");
            }
        }
    }

    /**
     * Acquire Decode capacity and commit one exact queue replacement. Missing
     * victims never degrade to a plain offer, and planning owns no capacity.
     */
    public QueueReplacementCommit commitReplacingQueuedVictims(
            BalanceContext context,
            CompletableFuture<Response> future,
            RequestRegistry lifecycle,
            List<ScheduledRequest> exactVictims) {
        OwnedRoute route = requireOwned();
        blockedEndpoint = null;
        if (!tryReserveDecode(context, route)) {
            return new QueueReplacementCommit(
                    WorkerBatcher.QueueReplacementStatus.DECLINED, null);
        }
        ScheduledRequest item = buildItem(
                context, future, System.currentTimeMillis());
        context.setRouteSubmittedNanos(System.nanoTime());
        WorkerBatcher.QueueReplacementStatus status =
                commitPreparedReplacement(
                        lifecycle, item, exactVictims);
        return new QueueReplacementCommit(
                status,
                status == WorkerBatcher.QueueReplacementStatus.SUCCESS
                        ? item : null);
    }

    private WorkerBatcher.QueueReplacementStatus commitPreparedReplacement(
            RequestRegistry lifecycle,
            ScheduledRequest item,
            List<ScheduledRequest> exactVictims) {
        OwnedRoute route = requireOwned();
        WorkerBatcher.QueueReplacementStatus[] replacement =
                new WorkerBatcher.QueueReplacementStatus[1];
        boolean committed = commitPublication(
                route,
                lifecycle,
                item,
                () -> {
                    WorkerBatcher.QueueReplacementStatus result =
                            route.prefillEndpoint().replaceQueued(
                                    route.prefillPin(),
                                    exactVictims,
                                    item);
                    replacement[0] = result;
                    return result == WorkerBatcher.QueueReplacementStatus.SUCCESS;
                }) == PlacementResult.Status.SUCCESS;
        return resolveReplacement(committed, replacement[0]);
    }

    private static boolean tryOfferPinned(OwnedRoute route, ScheduledRequest item) {
        return route.prefillEndpoint().offerPinned(
                route.prefillPin(), item);
    }

    private static WorkerBatcher.QueueReplacementStatus resolveReplacement(
            boolean committed,
            WorkerBatcher.QueueReplacementStatus replacement) {
        if (replacement == null) {
            if (committed) {
                throw new IllegalStateException(
                        "ACTIVE publication committed without replacement");
            }
            return WorkerBatcher.QueueReplacementStatus.CONFLICT;
        }
        if (committed != (replacement == WorkerBatcher.QueueReplacementStatus.SUCCESS)) {
            throw new IllegalStateException(
                    "queue replacement and ACTIVE publication disagree: " + replacement);
        }
        return replacement;
    }

    private void finishCommitted(OwnedRoute expected) {
        if (ownedRoute != expected) {
            throw new IllegalStateException(
                    "queue route admission ownership changed during commit");
        }
        // RequestSlot/ScheduledRequest now owns the exact Decode reservation. Clear
        // the admission owner before exact no-fail pin release.
        ownedRoute = null;
        expected.prefillPin().close();
        if (expected.decodePin() != null) {
            expected.decodePin().close();
        }
    }

    private OwnedRoute requireOwned() {
        OwnedRoute route = ownedRoute;
        if (route == null) {
            throw new IllegalStateException(
                    "queue route admission was already consumed");
        }
        return route;
    }

    @Override
    public void close() {
        OwnedRoute route = ownedRoute;
        ownedRoute = null;
        if (route == null) {
            return;
        }
        WorkerEndpoint.GenerationPin prefillPin = route.prefillPin();
        WorkerEndpoint.GenerationPin decodePin = route.decodePin();
        try (prefillPin; decodePin) {
            if (route.decodeEndpoint() != null
                    && route.decodeReservation() != null) {
                route.decodeEndpoint().releaseReservationExact(
                        route.decodeReservation());
            }
        }
    }

    private record OwnedRoute(
            PrefillEndpoint prefillEndpoint,
            WorkerEndpoint.GenerationPin prefillPin,
            ServerStatus prefillStatus,
            long prefillPlacementVersion,
            DecodeEndpoint decodeEndpoint,
            WorkerEndpoint.GenerationPin decodePin,
            DecodeEndpoint.ReservationHandle decodeReservation,
            ServerStatus decodeStatus,
            long decodePlacementVersion) {

        private OwnedRoute withDecodeReservation(
                DecodeEndpoint.ReservationHandle reservation) {
            return new OwnedRoute(
                    prefillEndpoint,
                    prefillPin,
                    prefillStatus,
                    prefillPlacementVersion,
                    decodeEndpoint,
                    decodePin,
                    reservation,
                    decodeStatus,
                    decodePlacementVersion);
        }
    }
}
