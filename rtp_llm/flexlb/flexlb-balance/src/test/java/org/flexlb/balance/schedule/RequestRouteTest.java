package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.DebugInfo;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;

/**
 * Checks frozen role metadata and endpoint validation without making route validation own endpoint cleanup.
 */
class RequestRouteTest {
    @Test
    void routeReadsFrozenSelectionFacts() {
        var context = context();
        var worker = prefill();
        var selectedEndpoint = worker;
        var metadata = metadata(worker);
        metadata.setDpRank(7L);
        var debug = new DebugInfo();
        debug.setHitCacheLen(17L);
        metadata.setDebugInfo(debug);
        try {
            var result =
                    RoutingResult.success(
                            new RequestResult(
                                    context,
                                    List.of(
                                            SelectionTestSupport.metadata(
                                                    selectedEndpoint, metadata, 10L))));
            {
                var route = result.value();
                metadata.setGroup("changed");
                metadata.setDpRank(99L);
                debug.setHitCacheLen(99L);
                route.role(org.flexlb.config.service.RoleType.PREFILL)
                        .toServerStatus(route.requestId())
                        .setGroup("export changed");
                route.successResponse()
                        .getServerStatus()
                        .getFirst()
                        .getDebugInfo()
                        .setHitCacheLen(1L);
                assertEquals(
                        "original",
                        route.role(org.flexlb.config.service.RoleType.PREFILL)
                                .toServerStatus(route.requestId())
                                .getGroup());
                assertEquals(17L, route.role(RoleType.PREFILL).hitCacheTokens());
                assertEquals(
                        7L,
                        route.role(RoleType.PREFILL).toServerStatus(route.requestId()).getDpRank());
                assertEquals(10L, route.computeTokens());
                assertSame(selectedEndpoint, route.prefillEndpoint());
                assertSame(worker, route.prefillEndpoint());
                assertTrue(!selectedEndpoint.hasStoppedAcceptingRequests());
            }
            assertFalse(selectedEndpoint.hasStoppedAcceptingRequests());
            assertEquals(
                    "original",
                    result.value()
                            .role(org.flexlb.config.service.RoleType.PREFILL)
                            .toServerStatus(result.value().requestId())
                            .getGroup());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(worker);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"failed", "request", "work", "role"})
    void validationFailureLeavesEndpointAvailable(String invalid) {
        var worker = prefill();
        var selectedEndpoint = worker;
        var metadata = metadata(worker);
        if (invalid.equals("failed")) {
            metadata.setSuccess(false);
        }
        if (invalid.equals("request")) {
            metadata.setRequestId(String.valueOf(99L));
        }
        if (invalid.equals("role")) {
            metadata.setRole(RoleType.DECODE);
        }
        try {
            var route = new java.util.ArrayList<RoleSelection>();
            assertThrows(
                    RuntimeException.class,
                    () -> {
                        route.add(
                                SelectionTestSupport.metadata(
                                        selectedEndpoint,
                                        metadata,
                                        invalid.equals("work") ? -1L : 10L));
                        new RequestResult(context(), route);
                    });
            assertFalse(selectedEndpoint.hasStoppedAcceptingRequests());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(worker);
        }
    }

    @Test
    void duplicateRoleRejectsRouteWithoutRetiringEndpoint() {
        var worker = prefill();
        var first = worker;
        var duplicate = worker;
        try {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            new RequestResult(
                                    context(),
                                    List.of(
                                            SelectionTestSupport.metadata(
                                                    first, metadata(worker), 1L),
                                            SelectionTestSupport.metadata(
                                                    duplicate, metadata(worker), 1L))));
            assertFalse(first.hasStoppedAcceptingRequests());
            assertFalse(duplicate.hasStoppedAcceptingRequests());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(worker);
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = RoleType.class,
            names = {"PREFILL", "PDFUSION", "DECODE", "ENCODER", "VIT"})
    void retirementBeforeOrAfterAddInvalidatesSelectionForEveryRole(RoleType role) {
        for (boolean retireBeforeAdd : List.of(false, true)) {
            var status =
                    WorkerStatus.createDiscovered(role, "original", "chosen", 8081, 8091, null);
            WorkerEndpoint worker =
                    switch (role) {
                        case PREFILL, PDFUSION ->
                                org.flexlb.worker.endpoint.EndpointTestSupport.realPrefill(
                                        status,
                                        SchedulingTestConfig.newConfig(),
                                        mock(EngineCancellation.class),
                                        new RequestRepository(),
                                        mock(DeliveryMetricsReporter.class));
                        case DECODE ->
                                org.flexlb.worker.endpoint.EndpointTestSupport.decode(
                                        status, new RequestRepository());
                        case ENCODER -> new EncoderEndpoint(status, new CapacityNotifier());
                        case VIT -> new WorkerEndpoint(status);
                        default -> throw new AssertionError(role);
                    };
            var prefill = prefill();
            try {
                var route = new java.util.ArrayList<RoleSelection>();
                if (role == RoleType.DECODE || role == RoleType.VIT)
                    route.add(SelectionTestSupport.metadata(prefill, metadata(prefill), 1L));
                var metadata = metadata(worker);
                if (retireBeforeAdd)
                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
                route.add(SelectionTestSupport.metadata(worker, metadata, 1L));
                if (!retireBeforeAdd)
                    org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
                var selected = new RequestResult(context(), route);
                var result = selected.validateEndpoints(selected);
                assertEquals(RoutingResult.Status.RETRY, result.status());
                assertNull(result.blocker());
                assertEquals(java.util.List.of(), result.roles());
            } finally {
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(prefill);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"PDV", "PVD", "DPV", "DVP", "VPD", "VDP"})
    void preparationPreservesSelectionOrderAndEndpointAvailability(String order) {
        var prefill = prefill();
        var decode =
                org.flexlb.worker.endpoint.EndpointTestSupport.decode(
                        WorkerStatus.createDiscovered(
                                RoleType.DECODE, "original", "decode", 8080, 8090, null),
                        new RequestRepository());
        var vit =
                new WorkerEndpoint(
                        WorkerStatus.createDiscovered(
                                RoleType.VIT, "original", "vit", 8080, 8090, null));
        var selectedWorkers = new java.util.ArrayList<WorkerEndpoint>();
        var roles = new java.util.ArrayList<RoleType>();
        try {
            var route = new java.util.ArrayList<RoleSelection>();
            for (char role : order.toCharArray()) {
                WorkerEndpoint worker =
                        switch (role) {
                            case 'P' -> prefill;
                            case 'D' -> decode;
                            default -> vit;
                        };
                var selectedEndpoint = worker;
                selectedWorkers.add(selectedEndpoint);
                roles.add(worker.getStatus().getRole());
                route.add(
                        SelectionTestSupport.metadata(
                                selectedEndpoint, metadata(worker), role == 'P' ? 1L : -1L));
            }
            var selected = new RequestResult(context(), route);
            assertEquals(
                    roles,
                    selected.successResponse().getServerStatus().stream()
                            .map(ServerStatus::getRole)
                            .toList());
            for (var selectedEndpoint : selectedWorkers) {
                assertFalse(selectedEndpoint.hasStoppedAcceptingRequests());
            }
            assertTrue(
                    selectedWorkers.stream()
                            .noneMatch(WorkerEndpoint::hasStoppedAcceptingRequests));
            assertEquals(
                    roles,
                    selected.successResponse().getServerStatus().stream()
                            .map(ServerStatus::getRole)
                            .toList());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(prefill);
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(decode);
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(vit);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(prefill);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(decode);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(vit);
        }
    }

    @Test
    void validationFailureDoesNotInvokeEndpointCleanup() {
        var worker = prefill();
        var selectedEndpoint = spy(worker);
        var metadata = metadata(worker);
        metadata.setSuccess(false);
        try {
            var route = new java.util.ArrayList<RoleSelection>();
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    route.add(
                                            SelectionTestSupport.metadata(
                                                    selectedEndpoint, metadata, 1L)));
            assertEquals(List.of(), List.of(failure.getSuppressed()));
            assertFalse(selectedEndpoint.hasStoppedAcceptingRequests());
        } finally {
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(worker);
            org.flexlb.worker.endpoint.EndpointTestSupport.awaitClosed(worker);
        }
    }

    private static RequestContext context() {
        return RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
    }

    private static PrefillEndpoint prefill() {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "original", "127.0.0.1", 8080, 8090, null);
        return org.flexlb.worker.endpoint.EndpointTestSupport.realPrefill(
                status,
                SchedulingTestConfig.newConfig(),
                mock(EngineCancellation.class),
                new RequestRepository(),
                mock(DeliveryMetricsReporter.class));
    }

    private static ServerStatus metadata(WorkerEndpoint worker) {
        var result =
                ServerStatus.forWorker(
                        worker.getStatus().getRole(),
                        String.valueOf(42L),
                        worker.getStatus().topologySnapshot(),
                        worker.getStatus().committedEngineObservation());
        return result;
    }
}
