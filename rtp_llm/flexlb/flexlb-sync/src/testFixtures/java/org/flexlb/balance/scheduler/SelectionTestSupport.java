package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;

import java.util.function.Function;

/**
 * Deterministic policy fixtures still execute the production registration callback.
 */
public final class SelectionTestSupport {
    private SelectionTestSupport() {}

    public static RoleSelection registeredDecode(
            DecodeEndpoint endpoint,
            ServerStatus status,
            DecodeResources.RequestReservation registration) {
        return new RoleSelection.Decode(endpoint, status, registration);
    }

    /**
     * Constructs frozen metadata for tests of response projection without registering resources.
     */
    public static RoleSelection metadata(
            WorkerEndpoint endpoint, ServerStatus status, long workMs) {
        return switch (status.getRole()) {
            case PREFILL, PDFUSION ->
                    new RoleSelection.Prefill((PrefillEndpoint) endpoint, status, workMs, null);
            case DECODE -> new RoleSelection.Decode((DecodeEndpoint) endpoint, status, null);
            default -> new RoleSelection.Auxiliary(endpoint, status);
        };
    }

    /**
     * Complete a deterministic strategy answer through the production registration callback.
     */
    public static <T> PlacementResult<T, PlacementKey> registerResult(
            org.mockito.invocation.InvocationOnMock call,
            PlacementResult<T, PlacementKey> selected) {
        if (selected.status() != PlacementResult.Status.SUCCESS) return selected;
        Function<T, PlacementResult.Registration> registration =
                call.getArgument(call.getArguments().length - 1);
        return switch (registration.apply(selected.value())) {
            case REGISTERED -> selected;
            case CONFLICT, UNAVAILABLE -> PlacementResult.retry();
            case BLOCKED -> {
                PlacementKey key;
                if (selected.value() instanceof PrefillSelectionStrategy.Selection<?> prefill)
                    key =
                            new PlacementKey(
                                    call.getArgument(2), call.getArgument(3), prefill.address());
                else {
                    var decode = (DecodeResources.DecodeRoutingView) selected.value();
                    key = new PlacementKey(RoleType.DECODE, call.getArgument(2), decode.address());
                }
                yield PlacementResult.blocked(
                        key, Response.error(StrategyErrorType.RESOURCE_EXHAUSTED), null);
            }
        };
    }

    /**
     * Register fixture work against the endpoint observation, with no request-count limit.
     * KV demand must still fit the full reported capacity.
     */
    public static DecodeResources.Acquisition<
                    DecodeResources.ReserveStatus, DecodeResources.RequestReservation>
            registerQueuedDecode(
                    DecodeEndpoint endpoint,
                    String id,
                    long hardKv,
                    long expectedKv,
                    int priority) {
        return endpoint.tryRegisterSelection(
                endpoint.routingView(),
                id,
                hardKv,
                expectedKv,
                priority,
                new DecodeResources.AdmissionCapacity(0L, 100L));
    }

    /**
     * Register endpoint work for a mocked execution collaborator in ledger tests.
     * Real executions must receive their registration in their RoleSelection constructor.
     */
    public static CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord> registerPrefill(
            PrefillEndpoint endpoint, RouteResult route, long workMs) {
        return registerPrefill(endpoint, route, workMs, false);
    }

    public static CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord> registerPrefill(
            PrefillEndpoint endpoint, RouteResult route, long workMs, long limit) {
        org.springframework.test.util.ReflectionTestUtils.setField(endpoint, "requestLimit", limit);
        return registerPrefill(endpoint, route, workMs, false);
    }

    public static CapacityBoundary.Attempt<PrefillEndpoint.RequestRecord> registerPrefill(
            PrefillEndpoint endpoint, RouteResult route, long workMs, boolean queued) {
        java.util.Map<?, ?> requests =
                (java.util.Map<?, ?>)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                endpoint, "requests");
        if (route.prefillReservation() != null
                && requests.get(route.requestId()) == route.prefillReservation())
            return queued
                    ? CapacityBoundary.Attempt.accepted(route.prefillReservation())
                    : CapacityBoundary.Attempt.rejected(CapacityBoundary.OWNERSHIP_LOST);
        if (!org.mockito.Mockito.mockingDetails(route).isMock())
            throw new IllegalArgumentException("Register before constructing a real RouteResult");
        var input =
                new GroupPlanner.Item(
                        route.requestId(),
                        route.priority(),
                        route.enqueueSeq(),
                        route.enqueuedAtMs(),
                        route.expiresAtMs(),
                        route.seqLen(),
                        route.hitCache());
        if (org.mockito.Mockito.mockingDetails(route).getMockCreationSettings().getDefaultAnswer()
                != org.mockito.Mockito.CALLS_REAL_METHODS)
            org.mockito.Mockito.doReturn(input).when(route).planningInput();
        PrefillEndpoint.Registration registered;
        do {
            registered =
                    endpoint.tryRegisterSelection(
                            input, endpoint.captureRouteProjectionInputs(), workMs, queued);
        } while (registered.status() == PlacementResult.Registration.CONFLICT);
        if (registered.status() != PlacementResult.Registration.REGISTERED)
            return CapacityBoundary.Attempt.rejected(
                    registered.status() == PlacementResult.Registration.UNAVAILABLE
                            ? CapacityBoundary.OWNERSHIP_LOST
                            : CapacityBoundary.unavailable(
                                    CapacityBoundary.Resource.PREFILL_REQUEST));
        var record = registered.record();
        record.bind(route);
        org.mockito.Mockito.doReturn(record).when(route).prefillReservation();
        return CapacityBoundary.Attempt.accepted(record);
    }
}
