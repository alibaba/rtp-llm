package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.SelectionResult;
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
        return decode(endpoint, status, registration);
    }

    public static RoleSelection decode(
            DecodeEndpoint endpoint,
            ServerStatus status,
            DecodeEndpoint.RequestRegistration registration) {
        initializeMockStatus(endpoint, status);
        return RoleSelection.decode(endpoint, topology(status), registration);
    }

    /**
     * Constructs frozen metadata for tests of response projection without registering resources.
     */
    public static RoleSelection metadata(
            WorkerEndpoint endpoint, ServerStatus status, long workMs) {
        return switch (status.getRole()) {
            case PREFILL, PDFUSION -> prefill((PrefillEndpoint) endpoint, status, workMs, null);
            case DECODE -> decode((DecodeEndpoint) endpoint, status, null);
            case ENCODER -> encoder((EncoderEndpoint) endpoint, status, null);
            default -> auxiliary(endpoint, status);
        };
    }

    public static RoleSelection prefill(
            PrefillEndpoint endpoint,
            ServerStatus status,
            long computeTokens,
            PrefillEndpoint.RequestRegistration registration) {
        initializeMockStatus(endpoint, status);
        return RoleSelection.prefill(
                endpoint,
                topology(status),
                status.getDebugInfo() == null ? 0L : status.getDebugInfo().getHitCacheLen(),
                computeTokens,
                registration);
    }

    public static RoleSelection encoder(
            EncoderEndpoint endpoint,
            ServerStatus status,
            EncoderEndpoint.RequestRegistration registration) {
        initializeMockStatus(endpoint, status);
        return RoleSelection.encoder(endpoint, topology(status), registration);
    }

    public static RoleSelection auxiliary(WorkerEndpoint endpoint, ServerStatus status) {
        initializeMockStatus(endpoint, status);
        return RoleSelection.auxiliary(endpoint, topology(status));
    }

    /**
     * Supplies a deterministic pure choice; production selection performs admission and registration.
     */
    public static <T> java.util.Optional<T> selectionResult(
            org.mockito.invocation.InvocationOnMock call,
            SelectionResult<T, CapacityKey> selected) {
        if (selected.status() != SelectionResult.Status.SUCCESS) return java.util.Optional.empty();
        if (selected.value() instanceof CostBasedPrefillStrategy.Selection value) {
            java.util.List<PrefillEndpoint> inputs = call.getArgument(2);
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
                id,
                endpoint.selectionSnapshot(),
                hardKv,
                expectedKv,
                priority,
                new DecodeResources.AdmissionCapacity(0L, 100L));
    }

    public static WorkerEndpoint.Registration<PrefillEndpoint.RequestRegistration> registerPrefill(
            PrefillEndpoint endpoint, RequestRoute route, long computeTokens) {
        var registered =
                endpoint.tryRegisterSelection(
                        route.requestId(), endpoint.selectionSnapshot(), computeTokens);
        if (registered.status() == WorkerEndpoint.RegistrationStatus.REGISTERED) {
            org.mockito.Mockito.doReturn(registered.value()).when(route).prefillRegistration();
        }
        return registered;
    }

    private static void initializeMockStatus(WorkerEndpoint endpoint, ServerStatus selected) {
        var mock = org.mockito.Mockito.mockingDetails(endpoint);
        if (!mock.isMock() || mock.isSpy()) return;
        var status = endpoint.getStatus();
        if (status == null) {
            status =
                    org.flexlb.worker.status.WorkerStatus.createDiscovered(
                            selected.getRole(),
                            selected.getGroup(),
                            java.util.Objects.requireNonNullElse(selected.getServerIp(), "127.0.0.1"),
                            selected.getHttpPort() > 0 ? selected.getHttpPort() : 8080,
                            selected.getGrpcPort() > 0 ? selected.getGrpcPort() : 8081,
                            null);
            org.mockito.Mockito.when(endpoint.getStatus()).thenReturn(status);
        } else if (org.mockito.Mockito.mockingDetails(status).isMock() && status.getRole() == null) {
            org.mockito.Mockito.when(status.getRole()).thenReturn(selected.getRole());
        }
    }

    private static org.flexlb.worker.status.WorkerStatus.TopologySnapshot topology(
            ServerStatus status) {
        return new org.flexlb.worker.status.WorkerStatus.TopologySnapshot(
                status.getGroup(),
                status.getServerIp(),
                status.getHttpPort(),
                status.getGrpcPort(),
                null,
                null,
                status.getRoutingEngineIndex(),
                (int)
                        org.springframework.test.util.ReflectionTestUtils.getField(
                                status, "routingMultiEngineNum"));
    }

}
