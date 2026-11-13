package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;

/**
 * Builds registered routes and controlled queue dependencies while leaving resource mutation to endpoint APIs.
 */
public final class RequestExecutionTestSupport {

    private RequestExecutionTestSupport() {}

    /**
     * Queue and ledger unit tests supply registration separately from their mocked route input.
     * An existing Decode registration may remain as an immutable collaborator; Prefill is bound
     * when the queue fixture registers its work. Selection integration tests use real executions.
     */
    public static RequestResult mockProtocolRoute(RequestResult metadata) {
        org.junit.jupiter.api.Assertions.assertNull(
                metadata.prefillRegistration(),
                "mock route input must not already own Prefill registration");
        return org.mockito.Mockito.mock(
                RequestResult.class,
                org.mockito.Mockito.withSettings()
                        .useConstructor(metadata.ctx(), metadata.roles())
                        .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
    }

    /**
     * Register Prefill before constructing a complete execution for delivery integration tests.
     */
    public static RequestResult registerPrefillRoute(
            RequestContext context,
            PrefillEndpoint prefill,
            DecodeEndpoint decode,
            DecodeEndpoint.RequestRegistration decodeRegistration,
            long workMs,
            long enqueuedAtMs) {
        if (context.getSchedulingMetadata() == null)
            context.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                            context.getRequest().getRawPriority(),
                            context.getConfig()
                                    .resolveExpiresAtMs(context.getTiming().startedAtMs())));
        var ledger = org.flexlb.worker.endpoint.EndpointTestSupport.resources(prefill);
        var registered =
                ledger.tryRegisterSelection(
                        context.getRequestId(), ledger.selectionSnapshot(), workMs);
        org.junit.jupiter.api.Assertions.assertEquals(
                WorkerEndpoint.RegistrationStatus.REGISTERED, registered.status());
        var roles = new java.util.ArrayList<RoleSelection>();
        try {
            roles.add(
                    RoleSelection.prefill(
                            prefill,
                            metadata(
                                    prefill,
                                    context.getRequestId(),
                                    org.flexlb.config.service.RoleType.PREFILL),
                            workMs,
                            registered.value()));
            if (decode != null)
                roles.add(
                        RoleSelection.decode(
                                decode,
                                metadata(
                                        decode,
                                        context.getRequestId(),
                                        org.flexlb.config.service.RoleType.DECODE),
                                decodeRegistration));
            return new RequestResult(context, roles);
        } catch (RuntimeException | Error failure) {
            org.flexlb.util.Failures.append(
                    failure,
                    org.flexlb.util.Failures.run(
                            null, () -> ledger.rollbackSelection(registered.value())));
            org.flexlb.util.Failures.append(
                    failure,
                    org.flexlb.util.Failures.run(
                            null,
                            () ->
                                    RoleSelection.releaseDecode(
                                            decode,
                                            decodeRegistration,
                                            WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK)));
            throw failure;
        }
    }

    private static org.flexlb.worker.status.ServerStatus metadata(
            WorkerEndpoint endpoint, String requestId, org.flexlb.config.service.RoleType role) {
        var status = new org.flexlb.worker.status.ServerStatus();
        status.setSuccess(true);
        status.setRequestId(requestId);
        status.setRole(role);
        status.setServerIp(endpoint.getIp());
        status.setHttpPort(endpoint.getHttpPort());
        status.setGrpcPort(endpoint.getGrpcPort());
        return status;
    }
}
