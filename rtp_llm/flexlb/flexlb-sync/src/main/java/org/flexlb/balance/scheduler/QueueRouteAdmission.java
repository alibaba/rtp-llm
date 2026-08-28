package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointGenerationRetiredException;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * Sole owner of a selected queue route before ACTIVE publication.
 *
 * <p>The object is thread-confined to the routing/scheduling call.  It owns
 * both endpoint-generation pins and Decode's exact provisional reservation.
 * A successful commit moves the reservation into the canonical RequestSlot /
 * BatchItem owner and releases the short route pins.  Closing an uncommitted
 * admission performs exact local rollback.</p>
 */
public final class QueueRouteAdmission implements AutoCloseable {

    private final long requestId;
    private final Response response;
    private OwnedRoute ownedRoute;

    private QueueRouteAdmission(
            long requestId, Response response, OwnedRoute ownedRoute) {
        this.requestId = requestId;
        this.response = Objects.requireNonNull(response, "response");
        this.ownedRoute = Objects.requireNonNull(ownedRoute, "ownedRoute");
    }

    static QueueRouteAdmission prepare(
            BalanceContext context,
            List<SelectedRole> selectedRoles,
            Response response) {
        long requestId = context.getRequestId();

        PrefillEndpoint prefillEndpoint = null;
        WorkerEndpoint.GenerationPin prefillPin = null;
        ServerStatus prefillStatus = null;
        DecodeEndpoint decodeEndpoint = null;
        WorkerEndpoint.GenerationPin decodePin = null;
        DecodeEndpoint.ReservationHandle decodeReservation = null;
        ServerStatus decodeStatus = null;
        long decodeExpectedKvTokens = 0L;

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
                    long sequenceLength = Math.max(
                            0L, context.getRequest().getSeqLen());
                    long expectedKv = context.getConfig()
                            .decodeKvReservationTokens(
                                    sequenceLength,
                                    context.getRequest().getMaxNewTokens(),
                                    selected.decodeTotalKv());
                    decodeEndpoint = decode;
                    decodePin = pin;
                    decodeReservation = decode.reserveQueuedPinned(
                            pin,
                            context.getRequestId(),
                            sequenceLength,
                            expectedKv,
                            context.getPriority());
                    decodeExpectedKvTokens = expectedKv;
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
                            decodeReservation,
                            // Queue admission reserves with the exact
                            // expected-demand estimate; the eviction
                            // hand-off path below has no local estimate.
                            decodeExpectedKvTokens,
                            decodeStatus));
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                if (decodeEndpoint != null && decodeReservation != null) {
                    decodeEndpoint.rollbackExact(decodeReservation);
                }
                throw failure;
            }
        }
    }

    /**
     * Move a Decode reservation produced by the asynchronous preemption
     * protocol into the ordinary queue-admission owner.  The fresh exact
     * Decode pin closes the retirement race between the typed preemption
     * result and ACTIVE publication; failure rolls the moved reservation back.
     */
    public static QueueRouteAdmission prepareExistingDecode(
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
            if (decodePin == null) {
                throw new EndpointGenerationRetiredException(
                        "Decode generation retired before queue admission");
            }
            decodeEndpoint.requirePinnedGeneration(decodePin);
            if (decodeEndpoint.markQueuedExact(decodePin, decodeReservation)
                    == DecodeEndpoint.MarkQueuedResult.NOT_OWNED) {
                throw new EndpointGenerationRetiredException(
                        "Decode reservation retired before queued handoff");
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
                            decodeReservation,
                            // The reservation was created by the eviction
                            // protocol; its expected-demand estimate stays
                            // engine-side (0 = unknown on the slot mirror).
                            0L,
                            decodeStatus));
        } catch (RuntimeException | Error failure) {
            WorkerEndpoint.GenerationPin ownedPrefillPin = prefillPin;
            WorkerEndpoint.GenerationPin ownedDecodePin = decodePin;
            try (ownedPrefillPin; ownedDecodePin) {
                decodeEndpoint.rollbackExact(decodeReservation);
                throw failure;
            }
        }
    }

    public Response response() {
        return response;
    }

    public BatchItem buildItem(
            BalanceContext context,
            CompletableFuture<Response> future,
            long enqueuedAtMs) {
        if (context.getRequestId() != requestId) {
            throw new IllegalArgumentException(
                    "queue admission cannot build another request");
        }
        OwnedRoute route = requireOwned();
        return new BatchItem(
                context,
                future,
                response,
                RequestLifecycleCoordinator.copyOf(route.prefillStatus()),
                RequestLifecycleCoordinator.copyOf(route.decodeStatus()),
                route.prefillEndpoint(),
                route.decodeEndpoint(),
                route.decodeReservation(),
                route.decodeExpectedKvTokens(),
                enqueuedAtMs);
    }

    /**
     * Atomically bind the exact item and publish it ACTIVE.  Pins are released
     * only after the registrar has dropped the slot monitor and WorkerBatcher
     * has dropped its queue lock.
     */
    public boolean commitTo(
            InflightCommitPort registrar,
            BatchItem item,
            boolean priorityAdmission) {
        OwnedRoute route = requireOwned();
        return commitPublication(
                route,
                registrar,
                item,
                priorityAdmission,
                () -> tryOfferPinned(route, item));
    }

    /** Execute one admission-owned endpoint mutation and ACTIVE publication. */
    private boolean commitPublication(
            OwnedRoute route,
            InflightCommitPort registrar,
            BatchItem item,
            boolean priorityAdmission,
            InflightCommitPort.ActivePublication activePublication) {
        if (requireOwned() != route) {
            throw new IllegalStateException(
                    "queue admission ownership changed before publication");
        }
        boolean committed = registrar.commitInflight(
                item,
                priorityAdmission,
                activePublication);
        if (committed) {
            finishCommitted(route);
        }
        return committed;
    }

    /** Commit one exact queue replacement; missing victims never degrade to a plain offer. */
    public ReplacementCommit commitReplacingQueuedVictims(
            InflightCommitPort registrar,
            BatchItem item,
            boolean priorityAdmission,
            List<DeliveryItem> exactVictims) {
        OwnedRoute route = requireOwned();
        PreparedReplacement prepared = PreparedReplacement.prepare(exactVictims);
        PrefillGenerationRuntime.QueueReplacement[] replacement =
                new PrefillGenerationRuntime.QueueReplacement[1];
        boolean committed = commitPublication(
                route,
                registrar,
                item,
                priorityAdmission,
                () -> {
                    PrefillGenerationRuntime.QueueReplacement result =
                            route.prefillEndpoint().replaceQueued(
                                    route.prefillPin(),
                                    prepared.deliveryItems(),
                                    item);
                    replacement[0] = result;
                    return result.status()
                            == PrefillGenerationRuntime.QueueReplacementStatus.SUCCESS;
                });
        return prepared.resolve(committed, replacement[0]);
    }

    private static boolean tryOfferPinned(OwnedRoute route, BatchItem item) {
        return route.prefillEndpoint().offerPinned(
                route.prefillPin(), item);
    }

    public enum ReplacementStatus {
        SUCCESS,
        CONFLICT,
        DECLINED,
        NOT_ATTEMPTED
    }

    /** Domain result of one admission-owned queue replacement publication. */
    public record ReplacementCommit(ReplacementStatus status) {
    }

    /**
     * Scheduler exact-type boundary. All result containers are allocated
     * before the endpoint can atomically replace queue ownership.
     */
    private record PreparedReplacement(
            List<DeliveryItem> deliveryItems,
            ReplacementCommit success,
            ReplacementCommit conflict,
            ReplacementCommit declined,
            ReplacementCommit notAttempted) {

        private static PreparedReplacement prepare(
                List<DeliveryItem> exactVictims) {
            return new PreparedReplacement(
                    exactVictims,
                    new ReplacementCommit(ReplacementStatus.SUCCESS),
                    new ReplacementCommit(ReplacementStatus.CONFLICT),
                    new ReplacementCommit(ReplacementStatus.DECLINED),
                    new ReplacementCommit(ReplacementStatus.NOT_ATTEMPTED));
        }

        private ReplacementCommit resolve(
                boolean committed,
                PrefillGenerationRuntime.QueueReplacement replacement) {
            if (replacement == null) {
                if (committed) {
                    throw new IllegalStateException(
                            "ACTIVE publication committed without replacement");
                }
                return notAttempted;
            }
            return switch (replacement.status()) {
                case SUCCESS -> {
                    if (!committed) {
                        throw new IllegalStateException(
                                "queue replacement committed without ACTIVE publication");
                    }
                    yield success;
                }
                case CONFLICT -> {
                    if (committed) {
                        throw new IllegalStateException(
                                "conflicting replacement published ACTIVE");
                    }
                    yield conflict;
                }
                case DECLINED -> {
                    if (committed) {
                        throw new IllegalStateException(
                                "declined replacement published ACTIVE");
                    }
                    yield declined;
                }
            };
        }
    }

    private void finishCommitted(OwnedRoute expected) {
        if (ownedRoute != expected) {
            throw new IllegalStateException(
                    "queue route admission ownership changed during commit");
        }
        // RequestSlot/BatchItem now owns the exact Decode reservation. Clear
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
                route.decodeEndpoint().rollbackExact(
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
            long decodeExpectedKvTokens,
            ServerStatus decodeStatus) {
    }
}
