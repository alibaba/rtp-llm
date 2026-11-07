package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;

class RequestRouteTest {
    @Test
    void executionReadsFrozenSelectionFacts() {
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
                    SchedulingTestConfig.prepareRoute(
                            context,
                            route -> {
                                route.add(selectedEndpoint, metadata, 10L, 5L);
                                return PlacementResult.success(route);
                            });
            {
                var route = result.value();
                metadata.setGroup("changed");
                metadata.setDpRank(99L);
                debug.setHitCacheLen(99L);
                route.prefill().setGroup("export changed");
                route.successResponse(true)
                        .getServerStatus()
                        .getFirst()
                        .getDebugInfo()
                        .setHitCacheLen(1L);
                assertEquals("original", route.prefill().getGroup());
                assertEquals(17L, route.hitCache());
                assertEquals(7L, route.prefillDpRank());
                assertEquals(10L, route.prefillWorkMs());
                assertSame(selectedEndpoint, route.prefillEndpoint());
                assertSame(worker, route.prefillEndpoint());
                assertTrue(!selectedEndpoint.isRetired());
            }
            assertFalse(selectedEndpoint.isRetired());
            assertEquals("original", result.value().prefill().getGroup());
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"failed", "request", "address", "work", "version", "role"})
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
        if (invalid.equals("address")) {
            metadata.setHttpPort(9999);
        }

        if (invalid.equals("role")) {
            metadata.setRole(RoleType.DECODE);
        }
        try {
            var route = new RouteResult.Builder(context());
            assertThrows(
                    RuntimeException.class,
                    () ->
                            route.add(
                                    selectedEndpoint,
                                    metadata,
                                    invalid.equals("work") ? -1L : 10L,
                                    invalid.equals("version") ? -1L : 0L));
            assertFalse(selectedEndpoint.isRetired());
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
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
                            SchedulingTestConfig.prepareRoute(
                                    context(),
                                    route -> {
                                        route.add(first, metadata(worker), 1L, 0L);
                                        route.add(duplicate, metadata(worker), 1L, 0L);
                                        return PlacementResult.success(route);
                                    }));
            assertFalse(first.isRetired());
            assertFalse(duplicate.isRetired());
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
        }
    }

    @Test
    void blockedResultLeavesSelectedEndpointsAvailable() {
        var worker = prefill();
        var selectedEndpoint = worker;
        try {
            var result =
                    SchedulingTestConfig.prepareRoute(
                            context(),
                            route -> {
                                route.add(selectedEndpoint, metadata(worker), 1L, 0L);
                                return PlacementResult.blocked(
                                        new PlacementKey(RoleType.DECODE, null, null));
                            });
            assertEquals(PlacementResult.Status.BLOCKED, result.status());
            assertFalse(selectedEndpoint.isRetired());
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
        }
    }

    @Test
    void failedAddCannotBeRecoveredIntoACompletedRoute() {
        var worker = prefill();
        var invalid = metadata(worker);
        invalid.setSuccess(false);
        try {
            var route = new RouteResult.Builder(context());
            assertThrows(IllegalArgumentException.class, () -> route.add(worker, invalid, 1L, 0L));
            var nextWorker = worker;
            assertThrows(
                    IllegalStateException.class,
                    () -> route.add(nextWorker, metadata(worker), 1L, 0L));
            assertFalse(nextWorker.isRetired());
            assertThrows(IllegalStateException.class, route::prepare);
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
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
                                org.flexlb.balance.endpoint.EndpointTestSupport.realPrefill(
                                        status,
                                        SchedulingTestConfig.newConfig(),
                                        mock(DeliveryDispatcher.class),
                                        new RequestRepository(),
                                        mock(DeliveryMetricsReporter.class));
                        case DECODE ->
                                org.flexlb.balance.endpoint.EndpointTestSupport.decode(
                                        status, new RequestRepository());
                        case ENCODER ->
                                new org.flexlb.balance.endpoint.EncoderEndpoint(
                                        status, new PlacementAvailability());
                        case VIT -> new WorkerEndpoint(status);
                        default -> throw new AssertionError(role);
                    };
            var prefill = prefill();
            try {
                var route = new RouteResult.Builder(context());
                if (role == RoleType.DECODE || role == RoleType.VIT)
                    route.add(prefill, metadata(prefill), 1L, 0L);
                var metadata = metadata(worker);
                if (retireBeforeAdd) worker.close();
                assertSame(metadata, route.add(worker, metadata, 1L, 0L));
                if (!retireBeforeAdd) worker.close();
                var selected = route.prepare();
                var result = route.validateEndpoints(PlacementResult.success(selected));
                assertEquals(PlacementResult.Status.ENDPOINT_CHANGED, result.status());
                assertEquals(role, result.blocker().role());
                assertEquals(selected.roles(), ((RouteResult) result.selection().value()).roles());
            } finally {
                worker.close();
                prefill.close();
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"PDV", "PVD", "DPV", "DVP", "VPD", "VDP"})
    void preparationPreservesSelectionOrderAndEndpointAvailability(String order) {
        var prefill = prefill();
        var decode =
                org.flexlb.balance.endpoint.EndpointTestSupport.decode(
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
            var route = new RouteResult.Builder(context());
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
                route.add(selectedEndpoint, metadata(worker), role == 'P' ? 1L : -1L, 0L);
            }
            var selected = route.prepare();
            assertEquals(
                    roles,
                    selected.successResponse(true).getServerStatus().stream()
                            .map(ServerStatus::getRole)
                            .toList());
            for (var selectedEndpoint : selectedWorkers) {
                assertFalse(selectedEndpoint.isRetired());
            }
            assertThrows(IllegalStateException.class, route::prepare);
            assertTrue(selectedWorkers.stream().noneMatch(WorkerEndpoint::isRetired));
            assertEquals(
                    roles,
                    selected.successResponse(true).getServerStatus().stream()
                            .map(ServerStatus::getRole)
                            .toList());
        } finally {
            prefill.close();
            decode.close();
            vit.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(prefill);
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(decode);
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(vit);
        }
    }

    @Test
    void validationFailureDoesNotInvokeEndpointCleanup() {
        var worker = prefill();
        var selectedEndpoint = spy(worker);
        var cleanup = new IllegalStateException("selectedEndpoint cleanup failed");
        doAnswer(
                        call -> {
                            call.callRealMethod();
                            throw cleanup;
                        })
                .when(selectedEndpoint)
                .close();
        var metadata = metadata(worker);
        metadata.setSuccess(false);
        try {
            var route = new RouteResult.Builder(context());
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> route.add(selectedEndpoint, metadata, 1L, 0L));
            assertEquals(List.of(), List.of(failure.getSuppressed()));
            assertFalse(selectedEndpoint.isRetired());
        } finally {
            worker.close();
            org.flexlb.balance.endpoint.EndpointTestSupport.awaitRetirement(worker);
        }
    }

    private static RequestContext context() {
        return RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 42L);
    }

    private static PrefillEndpoint prefill() {
        var status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "original", "127.0.0.1", 8080, 8090, null);
        return org.flexlb.balance.endpoint.EndpointTestSupport.realPrefill(
                status,
                SchedulingTestConfig.newConfig(),
                mock(DeliveryDispatcher.class),
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
