package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;

/**
 * Provides deterministic choices while the production selector performs registration.
 */
public final class SelectionTestSupport {
    private SelectionTestSupport() {}

    public static RoleSelection registeredDecode(
            DecodeEndpoint endpoint,
            ServerStatus status,
            DecodeEndpoint.RequestRegistration registration) {
        return RoleSelection.decode(endpoint, status, registration);
    }

    /**
     * Constructs frozen metadata for tests of response projection without registering resources.
     */
    public static RoleSelection metadata(
            WorkerEndpoint endpoint, ServerStatus status, long workMs) {
        return switch (status.getRole()) {
            case PREFILL, PDFUSION ->
                    RoleSelection.prefill((PrefillEndpoint) endpoint, status, workMs, null);
            case DECODE -> RoleSelection.decode((DecodeEndpoint) endpoint, status, null);
            case ENCODER -> RoleSelection.encoder((EncoderEndpoint) endpoint, status, null);
            default -> RoleSelection.auxiliary(endpoint, status);
        };
    }

    /**
     * Supplies a deterministic pure choice; production selection performs admission and registration.
     */
    public static <T> java.util.Optional<T> selectionResult(
            org.mockito.invocation.InvocationOnMock call, RoutingResult<T, CapacityKey> selected) {
        if (selected.status() != RoutingResult.Status.SUCCESS) return java.util.Optional.empty();
        if (selected.value() instanceof CostBasedPrefillStrategy.Selection value) {
            java.util.List<PrefillEndpoint> inputs = call.getArgument(1);
            var input =
                    inputs.stream()
                            .filter(
                                    candidate ->
                                            candidate == value.input()
                                                    || value.address().equals(candidate.ipPort()))
                            .findFirst()
                            .orElse(value.input());
            return (java.util.Optional<T>)
                    java.util.Optional.of(
                            new CostBasedPrefillStrategy.Selection(
                                    value.address(),
                                    value.generationId(),
                                    value.computeTokens(),
                                    value.cacheHitTokens(),
                                    value.routingCacheMatchTokens(),
                                    value.maximumRoutingCacheMatchTokens(),
                                    value.affinityReason(),
                                    input,
                                    value.cacheMatches(),
                                    value.selectionSnapshot()));
        }
        return java.util.Optional.of(selected.value());
    }

    /**
     * Register fixture work against the endpoint observation, with no request-count limit.
     * KV demand must still fit the full reported capacity.
     */
    public static WorkerEndpoint.Registration<DecodeEndpoint.RequestRegistration>
            registerQueuedDecode(
                    DecodeEndpoint endpoint,
                    String id,
                    long hardKv,
                    long expectedKv,
                    int priority) {
        return endpoint.tryRegisterSelection(
                endpoint.selectionSnapshot(),
                id,
                hardKv,
                expectedKv,
                priority,
                new DecodeResources.AdmissionCapacity(0L, 100L));
    }

    public static WorkerEndpoint.Registration<PrefillEndpoint.RequestRegistration> registerPrefill(
            PrefillEndpoint endpoint, RequestResult route, long computeTokens) {
        var registered =
                endpoint.tryRegisterSelection(
                        route.requestId(), endpoint.selectionSnapshot(), computeTokens);
        if (registered.status() == WorkerEndpoint.RegistrationStatus.REGISTERED) {
            registered.value().bind(route);
            org.mockito.Mockito.doReturn(registered.value()).when(route).prefillRegistration();
        }
        return registered;
    }
}
