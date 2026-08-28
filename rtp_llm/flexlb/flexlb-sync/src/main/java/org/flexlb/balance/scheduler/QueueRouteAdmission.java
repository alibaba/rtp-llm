package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.WorkerBatcher;
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
 * <p>The object is thread-confined to the routing/scheduling call.  It owns
 * both endpoint-generation pins. Exact Decode capacity is acquired only in
 * the short publication transaction. A successful commit moves ownership to
 * the canonical RequestSlot / ScheduledRequest; closing rolls back local ownership.</p>
 */
public final class QueueRouteAdmission implements AutoCloseable {

    private final long requestId;
    private final Response response;
    private final ScheduledRequest.DecodeReselection decodeReselection;
    private final PlacementAvailability placementAvailability;
    private OwnedRoute ownedRoute;

    private QueueRouteAdmission(
            long requestId,
            Response response,
            OwnedRoute ownedRoute,
            ScheduledRequest.DecodeReselection decodeReselection,
            PlacementAvailability placementAvailability) {
        this.requestId = requestId;
        this.response = Objects.requireNonNull(response, "response");
        this.ownedRoute = Objects.requireNonNull(ownedRoute, "ownedRoute");
        this.decodeReselection = decodeReselection;
        this.placementAvailability = placementAvailability;
    }

    static QueueRouteAdmission prepare(
            BalanceContext context,
            List<SelectedRole> selectedRoles,
            Response response) {
        return prepare(context, selectedRoles, response, null, null);
    }

    static QueueRouteAdmission prepare(
            BalanceContext context,
            List<SelectedRole> selectedRoles,
            Response response,
            ScheduledRequest.DecodeReselection decodeReselection,
            PlacementAvailability placementAvailability) {
        long requestId = context.getRequestId();

        PrefillEndpoint prefillEndpoint = null;
        WorkerEndpoint.GenerationPin prefillPin = null;
        ServerStatus prefillStatus = null;
        DecodeEndpoint decodeEndpoint = null;
        WorkerEndpoint.GenerationPin decodePin = null;
        ServerStatus decodeStatus = null;

        try {
            for (SelectedRole selected : selectedRoles) {
                ServerStatus status = selected.serverStatus();
                if (status.getRequestId() != requestId) {
                    throw new IllegalStateException(
                            "selected role belongs to another request");
                }
                RoleType role = status.getRole();
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
                            decodeEndpoint,
                            decodePin,
                            null,
                            decodeStatus),
                    decodeReselection,
                    placementAvailability);
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                throw failure;
            }
        }
    }

    /**
     * Move a Decode reservation produced by the asynchronous preemption
     * protocol into the ordinary queue-admission owner.  The fresh exact
     * Decode pin closes the retirement race between the typed preemption
     * result and ACTIVE publication. A retired generation returns null after
     * rolling the moved reservation back.
     */
    static QueueRouteAdmission tryPrepareExistingDecode(
            SelectedRole prefillSelection,
            DecodeEndpoint decodeEndpoint,
            DecodeEndpoint.ReservationHandle decodeReservation,
            ServerStatus decodeStatus,
            Response response) {
        long requestId = decodeReservation.requestId();

        WorkerEndpoint.GenerationPin prefillPin = null;
        WorkerEndpoint.GenerationPin decodePin = null;
        try {
            ServerStatus prefillStatus = prefillSelection.serverStatus();
            if (prefillStatus.getRequestId() != requestId
                    || decodeStatus.getRequestId() != requestId) {
                throw new IllegalStateException(
                        "decode eviction statuses belong to another request");
            }
            if (decodeStatus.getRole() != RoleType.DECODE
                    || !Objects.equals(
                            decodeStatus.getServerIp(), decodeEndpoint.getIp())
                    || decodeStatus.getHttpPort()
                            != decodeEndpoint.getHttpPort()) {
                throw new IllegalStateException(
                        "decode eviction metadata does not match exact endpoint");
            }
            prefillPin = prefillSelection.takeGenerationPin();
            if (!(prefillPin.endpoint() instanceof PrefillEndpoint prefillEndpoint)
                    || prefillStatus.getRole() != RoleType.PREFILL
                            && prefillStatus.getRole() != RoleType.PDFUSION) {
                throw new IllegalStateException(
                        "decode eviction requires one exact Prefill selection");
            }
            decodePin = decodeEndpoint.tryPinGeneration();
            if (decodePin != null) {
                decodeEndpoint.requirePinnedGeneration(decodePin);
                if (decodeEndpoint.markQueuedExact(
                        decodePin, decodeReservation)) {
                    return new QueueRouteAdmission(
                            requestId,
                            response,
                            new OwnedRoute(
                                    prefillEndpoint,
                                    prefillPin,
                                    prefillStatus,
                                    decodeEndpoint,
                                    decodePin,
                                    decodeReservation,
                                    decodeStatus),
                            null,
                            null);
                }
            }
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                decodeEndpoint.releaseReservationExact(decodeReservation);
                throw failure;
            }
        }
        WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
        WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
        try (ownedPrefillPin; ownedDecodePin) {
            decodeEndpoint.releaseReservationExact(decodeReservation);
        }
        return null;
    }

    public Response response() {
        return response;
    }

    /** Prefill capacity domain contended by this exact selected route. */
    PlacementKey prefillPlacementKey() {
        return placementKey(requireOwned().prefillStatus());
    }

    /** Decode capacity domain, or null when this model has no Decode role. */
    PlacementKey decodePlacementKey() {
        ServerStatus status = requireOwned().decodeStatus();
        return status == null ? null : placementKey(status);
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
                enqueuedAtMs,
                decodeReselection,
                placementAvailability);
    }

    /**
     * Acquire exact capacity and publish once. A blocked result owns no
     * endpoint resource and is safe to retry through a fresh route selection.
     */
    public PlacementResult<ScheduledRequest, PlacementKey> tryPublish(
            BalanceContext context,
            CompletableFuture<Response> future,
            RequestRegistry lifecycle) {
        OwnedRoute route = requireOwned();
        if (!tryReserveDecode(context, route)) {
            return PlacementResult.blocked(
                    placementKey(route.decodeStatus()));
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
                false,
                () -> exact.prefillEndpoint().offerPinnedForPlacement(
                        exact.prefillPin(), item));
        return switch (committed) {
            case SUCCESS -> PlacementResult.success(item);
            case BLOCKED -> PlacementResult.blocked(
                    placementKey(exact.prefillStatus()));
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
            return false;
        }
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

    private static PlacementKey placementKey(ServerStatus status) {
        return new PlacementKey(status.getRole(), status.getGroup());
    }

    /**
     * Atomically bind the exact item and publish it ACTIVE.  Pins are released
     * only after the registrar has dropped the slot monitor and WorkerBatcher
     * has dropped its queue lock.
     */
    public boolean commitTo(
            RequestRegistry lifecycle,
            ScheduledRequest item,
            boolean priorityAdmission) {
        OwnedRoute route = requireOwned();
        return commitPublication(
                route,
                lifecycle,
                item,
                priorityAdmission,
                () -> tryOfferPinned(route, item))
                == PlacementResult.Status.SUCCESS;
    }

    /** Execute one admission-owned endpoint mutation and ACTIVE publication. */
    private PlacementResult.Status commitPublication(
            OwnedRoute route,
            RequestRegistry lifecycle,
            ScheduledRequest item,
            boolean priorityAdmission,
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
                priorityAdmission,
                lifecycleConfig.getMaxDeliveredNotAcceptedRequestsGlobal(),
                lifecycleConfig.getDeliveredNotAcceptedTimeoutMs(),
                activePublication);
        if (committed
                == PlacementResult.Status.SUCCESS) {
            finishCommitted(route);
        }
        return committed;
    }

    /** Commit one exact queue replacement; missing victims never degrade to a plain offer. */
    public WorkerBatcher.QueueReplacementStatus commitReplacingQueuedVictims(
            RequestRegistry lifecycle,
            ScheduledRequest item,
            boolean priorityAdmission,
            List<ScheduledRequest> exactVictims) {
        OwnedRoute route = requireOwned();
        WorkerBatcher.QueueReplacementStatus[] replacement =
                new WorkerBatcher.QueueReplacementStatus[1];
        boolean committed = commitPublication(
                route,
                lifecycle,
                item,
                priorityAdmission,
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
            DecodeEndpoint decodeEndpoint,
            WorkerEndpoint.GenerationPin decodePin,
            DecodeEndpoint.ReservationHandle decodeReservation,
            ServerStatus decodeStatus) {

        private OwnedRoute withDecodeReservation(
                DecodeEndpoint.ReservationHandle reservation) {
            return new OwnedRoute(
                    prefillEndpoint,
                    prefillPin,
                    prefillStatus,
                    decodeEndpoint,
                    decodePin,
                    reservation,
                    decodeStatus);
        }
    }
}
